import asyncio
import os
from dataclasses import dataclass
from typing import Protocol


@dataclass
class GenResult:
    text: str
    finish_reason: str


class Backend(Protocol):
    def generate(self, prompts: list[str], config) -> list[GenResult]:
        ...


class APIBackend:
    def generate(self, prompts: list[str], config) -> list[GenResult]:
        return asyncio.run(self._generate(prompts, config))

    async def _generate(self, prompts: list[str], config) -> list[GenResult]:
        max_retries = config.model.max_retries
        semaphore = asyncio.Semaphore(config.model.concurrency)
        pending = list(enumerate(prompts))
        results: list[GenResult | None] = [None] * len(prompts)
        for attempt in range(max_retries):
            gathered = await asyncio.gather(
                *(self._limited_call(prompt, config, semaphore) for _, prompt in pending),
                return_exceptions=True,
            )
            next_pending = []
            for (idx, prompt), result in zip(pending, gathered):
                if isinstance(result, Exception):
                    next_pending.append((idx, prompt))
                else:
                    results[idx] = result
            pending = next_pending
            if not pending:
                break
            await asyncio.sleep(2**attempt * 0.25)
        for idx, _ in pending:
            results[idx] = GenResult(text="", finish_reason="error")
        return [result for result in results if result is not None]

    async def _limited_call(self, prompt: str, config, semaphore: asyncio.Semaphore) -> GenResult:
        async with semaphore:
            return await self._call(prompt, config)

    async def _call(self, prompt: str, config) -> GenResult:
        import httpx

        payload = {
            "model": config.model.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": config.generation.max_tokens,
            "temperature": config.generation.temperature,
            "stop": config.generation.stop,
        }
        timeout = httpx.Timeout(120.0)
        async with httpx.AsyncClient(base_url=config.model.base_url, timeout=timeout) as client:
            response = await client.post("chat/completions", json=payload, headers=api_headers())
        response.raise_for_status()
        data = response.json()
        choice = data["choices"][0]
        finish_reason = "length" if choice.get("finish_reason") == "length" else "stop"
        return GenResult(text=choice["message"]["content"], finish_reason=finish_reason)


class VLLMBackend:
    def generate(self, prompts: list[str], config) -> list[GenResult]:
        from vllm import LLM, SamplingParams

        params = SamplingParams(
            max_tokens=config.generation.max_tokens,
            temperature=config.generation.temperature,
            stop=config.generation.stop,
            seed=config.generation.seed,
        )
        llm_kwargs = {
            "model": config.model.path,
            "tensor_parallel_size": config.model.tensor_parallel_size,
        }
        if config.model.data_parallel_size != 1:
            llm_kwargs["data_parallel_size"] = config.model.data_parallel_size
        if config.model.max_model_len is not None:
            llm_kwargs["max_model_len"] = config.model.max_model_len
        if config.model.gpu_memory_utilization is not None:
            llm_kwargs["gpu_memory_utilization"] = config.model.gpu_memory_utilization
        if config.model.dtype is not None:
            llm_kwargs["dtype"] = config.model.dtype
        if config.model.max_num_seqs is not None:
            llm_kwargs["max_num_seqs"] = config.model.max_num_seqs
        if config.model.enforce_eager is not None:
            llm_kwargs["enforce_eager"] = config.model.enforce_eager
        llm = LLM(**llm_kwargs)
        outputs = llm.generate(prompts, params)
        results = []
        for output in outputs:
            item = output.outputs[0]
            finish_reason = "length" if item.finish_reason == "length" else "stop"
            results.append(GenResult(text=item.text, finish_reason=finish_reason))
        return results


def build_backend(config) -> Backend:
    if config.model.backend == "api":
        return APIBackend()
    if config.model.backend == "vllm":
        return VLLMBackend()
    raise ValueError(f"unknown backend: {config.model.backend}")


def api_headers() -> dict[str, str]:
    key = os.environ.get("OPENAI_API_KEY")
    return {"Authorization": f"Bearer {key}"} if key else {}
