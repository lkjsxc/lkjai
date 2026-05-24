from .api import call_kimi_cli_jobs, kimi_cli_job
from .batch_process import call_with_retries, empty_batch_result, process_response
from .generate_ids import family_for_ordinal, split_for_ordinal
from .prompting import build_generation_messages, generation_constraints


def generate_batch(config, api_key, prompt_template, fixture_rows, quarantine, state, ordinal):
    split = split_for_ordinal(ordinal)
    family = family_for_ordinal(config, ordinal)
    batch_size = max(1, int(config.get("batch_documents", 1)))
    messages = build_generation_messages(config, prompt_template, fixture_rows, split, family, batch_size, ordinal)
    api_result = call_with_retries(api_key, config, messages)
    result = empty_batch_result(split)
    result["api_calls"] = api_result["api_calls"]
    response = api_result["response"]
    processed = process_response(
        config,
        api_key,
        messages,
        response,
        batch_size,
        split,
        family,
        ordinal,
        quarantine,
        state,
    )
    processed["api_calls"] += result["api_calls"]
    return processed


def generate_parallel_batches(
    config, api_key, prompt_template, fixture_rows, quarantine, state, ordinal, count
):
    jobs = []
    for offset in range(count):
        current = ordinal + offset
        split = split_for_ordinal(current)
        family = family_for_ordinal(config, current)
        batch_size = max(1, int(config.get("batch_documents", 1)))
        messages = build_generation_messages(
            config, prompt_template, fixture_rows, split, family, batch_size, current
        )
        jobs.append(
            {
                "ordinal": current,
                "split": split,
                "family": family,
                "batch_size": batch_size,
                "messages": messages,
                "job": kimi_cli_job(
                    f"kimi-{split}-{current:09d}",
                    current,
                    split,
                    family,
                    messages,
                    generation_constraints(config, current),
                ),
            }
        )
    if str(config.get("api_provider", "")) == "kimi-cli":
        responses = call_kimi_cli_jobs(api_key, config, [job["job"] for job in jobs])
    else:
        responses = [
            call_with_retries(api_key, config, job["messages"])["response"]
            for job in jobs
        ]
    aggregate = empty_batch_result("mixed")
    aggregate["batches"] = []
    for job, response in sorted(zip(jobs, responses), key=lambda item: item[0]["ordinal"]):
        result = process_response(
            config,
            api_key,
            job["messages"],
            response,
            job["batch_size"],
            job["split"],
            job["family"],
            job["ordinal"],
            quarantine,
            state,
        )
        result["api_calls"] += max(1, response.attempts or 1)
        aggregate["api_calls"] += result["api_calls"]
        aggregate["generated_rows"] += result["generated_rows"]
        aggregate["rejected_rows"] += result["rejected_rows"]
        aggregate["token_estimate"] += result["token_estimate"]
        aggregate["errors"].extend(result["errors"])
        aggregate["batches"].append(result)
        if result["stop_reason"]:
            aggregate["status"] = result["status"]
            aggregate["stop_reason"] = result["stop_reason"]
            break
    return aggregate
