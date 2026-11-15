"""Compare two ways to return a mean token distribution with identical JIT outputs."""
import json
import jax
import jax.numpy as jnp
import numpy as np
from optax import microbatching as mb


def loss(params, example):
    inputs, labels = example
    log_probs = jax.nn.log_softmax(inputs @ params)
    nll = -jnp.take_along_axis(log_probs, labels[:, None], axis=-1).mean()
    return nll, jnp.exp(log_probs)


def compiled(params, examples, streaming):
    options = {"aux_accumulator": mb.AccumulationType.MEAN} if streaming else {}
    micro = mb.micro_grad(loss, has_aux=True, keep_batch_dim=False,
                          microbatch_size=2, **options)

    @jax.jit
    def train_step(w, batch):
        grads, result = micro(w, batch)
        distribution = result.aux if streaming else result.aux.mean(0)
        return grads, result.values, distribution

    program = train_step.lower(params, examples).compile()
    output = jax.block_until_ready(program(params, examples))
    info = program.memory_analysis()
    return output, dict(temp_bytes=info.temp_size_in_bytes, output_bytes=info.output_size_in_bytes)


def main():
    rng = np.random.default_rng(937)
    batch, seq, hidden, vocab = 32, 64, 32, 4096
    params = jnp.asarray(rng.normal(0, 0.05, (hidden, vocab)).astype(np.float32))
    examples = (jnp.asarray(rng.normal(size=(batch, seq, hidden)).astype(np.float32)),
                jnp.asarray(rng.integers(vocab, size=(batch, seq), dtype=np.int32)))
    original, base_mem = compiled(params, examples, False)
    updated, new_mem = compiled(params, examples, True)
    for x, y in zip(original, updated):
        np.testing.assert_allclose(np.asarray(x), np.asarray(y), rtol=4e-6, atol=1e-7)
    assert base_mem["output_bytes"] == new_mem["output_bytes"]
    print(json.dumps(dict(jax=jax.__version__, devices=[str(d) for d in jax.devices()],
                          batch=batch, sequence=seq, vocab=vocab, microbatch=2,
                          concat_then_mean=base_mem, reduce_in_loop=new_mem)))


if __name__ == "__main__":
    main()
