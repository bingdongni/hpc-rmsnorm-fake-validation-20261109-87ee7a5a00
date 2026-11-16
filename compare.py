import json
import multiprocessing as mp
from contextlib import closing
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from unittest.mock import Mock

project, tree, device, revision = sys.argv[1:]
sys.path.insert(0, str(Path(tree).resolve()))
fixed = revision == "candidate"

def race(path, barrier, output, value):
    import aisuite as ai
    with closing(ai.SQLiteStateStore(path)) as store:
        current = store.load_state("demo")
        current.state.add_user_message(value)
        barrier.wait(25)
        try:
            store.save_state("demo", current.state, revision=current.revision)
        except ai.StateConflictError:
            output.put(False)
        else:
            output.put(True)

def agents():
    import aisuite as ai
    from tests.agents.helpers import chat_response
    assert hasattr(ai, "SQLiteStateStore") == fixed
    if not fixed:
        return {"sqlite_backend":False}
    client = ai.Client()
    client.chat.completions.create = Mock(side_effect=[chat_response("first"),chat_response("second")])
    agent = ai.Agent(name="assistant",model="openai:fixture")
    with TemporaryDirectory() as temp:
        path = str(Path(temp)/"runs.db")
        with closing(ai.SQLiteStateStore(path)) as store:
            ai.Runner.run_sync(agent,"Hello",client=client,thread_id="demo",state_store=store)
            assert store.load_state("demo").revision == 1
        with closing(ai.SQLiteStateStore(path)) as store:
            r = ai.Runner.continue_sync(agent,"Follow up",client=client,thread_id="demo",state_store=store)
            assert r.final_output == "second" and store.load_state("demo").revision == 2
        messages = client.chat.completions.create.call_args.kwargs["messages"]
        assert [m["content"] for m in messages] == ["Hello","first","Follow up"]
        ctx = mp.get_context("spawn")
        barrier, output = ctx.Barrier(2), ctx.Queue()
        workers = [ctx.Process(target=race,args=(path,barrier,output,i)) for i in ("left","right")]
        for proc in workers: proc.start()
        for proc in workers:
            proc.join(50)
            assert proc.exitcode == 0
        assert sorted(output.get(timeout=3) for _ in workers) == [False, True]
        with closing(ai.SQLiteStateStore(path)) as store:
            assert store.load_state("demo").revision == 3
    return {"sqlite_backend":True,"model":"offline fixture","reopened":True,"concurrent_writers":2,"committed":1,"stale":1}

def local_attention(torch):
    from local_attention.transformer import LocalTransformer,LocalMHA
    errors = {}
    for norm,scale in [(False,8.),(True,2.),(True,8.)]:
        net = LocalMHA(dim=32,heads=2,dim_head=16,window_size=8,causal=True,qk_rmsnorm=norm,qk_scale=scale).double().to(device).eval()
        inputs = torch.randn(2,17,32,device=device,dtype=torch.float64)
        with torch.no_grad():
            _, cache = net(inputs[:,:-1],return_cache=True)
            delta = (net(inputs[:,-1:],cache=cache)-net(inputs)[:,-1:]).abs().max().item()
        errors[str((norm,scale))]=delta
        assert delta < 1e-10 if fixed or not norm else delta > .01
    # Dynamic position bias lets us compare repeated decode independently of the upstream RoPE-cache issue.
    model = LocalTransformer(num_tokens=48,max_seq_len=64,dim=32,depth=2,heads=2,dim_head=16,
             local_attn_window_size=8,qk_rmsnorm=True,qk_scale=8.,use_dynamic_pos_bias=True,num_residual_streams=1).to(device).eval()
    tokens=torch.randint(0,48,(3,21),device=device)
    with torch.no_grad():
        _,cache=model(tokens[:,:11],return_cache=True)
        deltas=[]
        for t in range(12,22):
            decoded,cache=model(tokens[:,:t],cache=cache,return_cache=True)
            deltas.append((decoded[:,-1]-model(tokens[:,:t])[:,-1]).abs().max().item())
        greedy=model.generate(tokens[:,:11],5,temperature=0.,use_kv_cache=True)
        expected=model.generate(tokens[:,:11],5,temperature=0.,use_kv_cache=False)
    if fixed:
        assert max(deltas)<1e-5 and torch.equal(greedy,expected)
    else: assert max(deltas)>.01
    return {"mha_errors":errors,"decode_steps":len(deltas),"max_logit_error":max(deltas),"greedy_matches":bool(torch.equal(greedy,expected))}

def rotations(torch):
    from e3nn import o3
    torch.set_default_dtype(torch.float32)
    angles=[torch.rand(6,dtype=torch.float64,device=device,requires_grad=True) for _ in range(3)]
    reference=o3.angles_to_matrix(*angles)
    try:
        result=o3.Irreps("1x1o").D_from_matrix(reference)
    except RuntimeError as exc:
        assert not fixed and device=="cuda" and "device" in str(exc).lower(), exc
        return {"baseline_device_error":True}
    err=(result-reference).abs().max().item()
    if not fixed:
        assert err>1e-9
        return {"cartesian_error":err}
    assert err<1e-12 and result.device.type==device
    mask=torch.randn_like(reference)
    g1=torch.autograd.grad((o3.wigner_D(1,*angles)*mask).sum(),angles)
    g2=torch.autograd.grad((reference*mask).sum(),angles)
    grad=max((a-b).abs().max().item() for a,b in zip(g1,g2))
    assert grad < 1e-11
    second=o3.rand_matrix(6,dtype=torch.float64,device=device)
    ir=o3.Irrep("2e")
    compose=(ir.D_from_matrix(reference@second)-ir.D_from_matrix(reference)@ir.D_from_matrix(second)).abs().max().item()
    assert compose < 1e-11
    return dict(cartesian_error=err,max_angle_grad_error=grad,degree_two_composition_error=compose)

def names():
    from sigma.collection import SigmaCollection
    from sigma.validation import SigmaValidator
    from sigma.validators.core import validators
    from uuid import UUID

    records = []
    for idx, name in enumerate(["first", "summary", None, "first", "summary", None], 1):
        rule = dict(title=f"Local application event {idx}", id=str(UUID(int=idx)),
                    logsource={"category":"application"},
                    detection={"selection":{"Kind":"message"}, "condition":"selection"})
        if name is not None:
            rule["name"] = name
        records.append(rule)
    collection = SigmaCollection.from_dicts(records)
    known = ["identifier_uniqueness","duplicate_title"]
    assert ("duplicate_name" in validators) == fixed
    selected = known + (["duplicate_name"] if fixed else [])
    issues = SigmaValidator.from_dict({"validators": selected}, validators).validate_rules(iter(collection))
    assert sorted((issue.name,len(issue.rules)) for issue in issues) == ([("first",2),("summary",2)] if fixed else [])
    return {"duplicate_groups_reported":len(issues),"rules":len(records),"unnamed":2}

if __name__=="__main__":
    if project=="aisuite":
        result=agents()
    elif project=="sigma":
        result=names()
    else:
        import torch
        torch.manual_seed(129)
        torch.set_num_threads(1)
        if device=="cuda":
            assert "4060" in torch.cuda.get_device_name()
            print("Device",torch.cuda.get_device_name(),torch.__version__)
        result=(rotations if project=="e3nn" else local_attention)(torch)
    print("Compared",json.dumps({"project":project,"revision":revision,"device":device,**result}))
