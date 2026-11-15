"""Compare native CSV validation to independent stable dense retrieval rankings."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import tempfile

parser=argparse.ArgumentParser()
parser.add_argument("checkout",type=Path)
parser.add_argument("assets",type=Path)
opt=parser.parse_args()
root=opt.checkout.resolve()
assets=opt.assets.resolve()
sys.path.insert(0,str(root/"src"))
import numpy as np
import torch
from open_clip import create_model_and_transforms, create_task, get_model_traits, get_tokenizer
from open_clip_train.data import get_csv_dataset
from open_clip_train.params import parse_args
from open_clip_train.train import evaluate
from open_clip_train import metrics as mm

assert Path(mm.__file__).resolve().is_relative_to(root)
assert torch.cuda.is_available()
print("GPU",torch.cuda.get_device_name(),torch.__version__,flush=True)
manifest=json.loads(Path(__file__).with_name("fixture-manifest.json").read_text(encoding="utf8"))
for file,digest in manifest.items():
    with (assets/file).open("rb") as f:assert hashlib.file_digest(f,"sha256").hexdigest()==digest
cfg=json.loads((assets/"open_clip_config.json").read_text(encoding="utf8"))
with tempfile.TemporaryDirectory() as tmp:
    table=Path(tmp)/"val.tsv"
    with (assets/"captions.tsv").open(encoding="utf8",newline="") as f: rows=list(csv.DictReader(f,delimiter="\t"))
    with table.open("w",encoding="utf8",newline="") as f:
        w=csv.DictWriter(f,["filepath","title"],delimiter="\t")
        w.writeheader()
        for row in rows:w.writerow(dict(row, filepath=str(assets/row["filepath"])))
    assert len(rows)==40 and len(set(x["filepath"] for x in rows))==8
    name="local-dir:"+str(assets)
    model,_,preprocess=create_model_and_transforms(name,device="cuda",output_dict=True)
    tokenizer=get_tokenizer(name)
    args=parse_args(["--val-data",str(table),"--batch-size","7","--workers","0",
                     "--precision","fp32","--val-csv-image-id-key","filepath","--val-retrieval-chunk-size","3"])
    args.device="cuda";args.rank=args.local_rank=0;args.world_size=1;args.distributed=False
    args.distill=False;args.save_logs=False
    task=create_task(args,model=model)
    def dataset():
        return get_csv_dataset(args,preprocess,False,tokenizer=tokenizer,model_traits=get_model_traits(model))
    data=dataset()
    grouped=evaluate(task,{"val":data},1,args,tokenizer=tokenizer)
    args.val_csv_image_id_key=None
    diagonal=evaluate(task,{"val":dataset()},1,args,tokenizer=tokenizer)
    ims=[];texts=[];ids=[]
    with torch.inference_mode():
        for sample in data.dataloader:
            ids.extend(sample.pop("retrieval_image_id").tolist())
            output=task(task.prepare_batch(sample,torch.device("cuda"),None))
            ims.append(output["image_features"].float().cpu());texts.append(output["text_features"].float().cpu())
    idx=list(dict.fromkeys(ids))
    scores=torch.cat(ims)[[ids.index(k) for k in idx]] @ torch.cat(texts).T
    gt=np.asarray(ids)
    for direction,matrix,relevant in [("image_to_text",scores,gt[None,:]==np.array(idx)[:,None]),
                                      ("text_to_image",scores.T,gt[:,None]==np.array(idx)[None,:])]:
        indices=np.argsort(-matrix.numpy(),axis=1,kind="stable")
        ranks=np.take_along_axis(relevant,indices,axis=1).argmax(axis=1)
        oracle={"mean_rank":ranks.mean()+1,"median_rank":np.floor(np.median(ranks))+1,
                **{f"R@{k}":np.mean(ranks<k) for k in [1,5,10]}}
        for k,v in oracle.items():np.testing.assert_allclose(grouped[f"{direction}_{k}"],v)
        print(direction,"R@1 diagonal",diagonal[direction+"_R@1"],"grouped",grouped[direction+"_R@1"],"dense oracle",oracle["R@1"])
    assert grouped["num_samples"]==diagonal["num_samples"]==40
    np.testing.assert_allclose(grouped["clip_val_loss"],diagonal["clip_val_loss"],rtol=1e-6)
    assert grouped["text_to_image_R@1"]>diagonal["text_to_image_R@1"]
