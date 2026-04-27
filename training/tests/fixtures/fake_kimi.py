#!/usr/bin/env python3
import argparse
import hashlib
import json
import sys


def main():
    if "--help" in sys.argv:
        print("Usage: kimi --quiet --print --final-message-only -p --input-format --output-format")
        return
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--print", action="store_true")
    parser.add_argument("--final-message-only", action="store_true")
    parser.add_argument("-p", "--prompt", default="")
    args, _ = parser.parse_known_args()
    prompt = args.prompt or sys.stdin.read()
    seed = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:8]
    mode = "sft" if "SFT" in prompt or "messages, tags, meta" in prompt else "pretrain"
    docs = 3
    for token in prompt.split():
        if token.isdigit():
            docs = int(token)
            break
    for index in range(docs):
        if mode == "pretrain":
            print(json.dumps(pretrain(index, seed), ensure_ascii=False))
        else:
            print(json.dumps(sft(index, seed), ensure_ascii=False))


def pretrain(index, seed):
    text = (
        f"This original synthetic lesson {seed}-{index} explains a practical idea with compact details. "
        "It defines the concept, gives a small example, and closes with a useful caution. "
        "No chat transcript is used, and the document is intended for full next-token learning."
    )
    return {
        "id": f"fake-pretrain-{seed}-{index:04d}",
        "mode": "pretrain",
        "language": "en",
        "domain": "education",
        "difficulty": "introductory",
        "title": f"Fake Pretrain {index}",
        "text": text,
        "metadata": {
            "source": "kimi_synthetic",
            "mode": "pretrain",
            "generated_at": "test",
            "prompt_version": "v1",
            "estimated_tokens": 80,
            "provenance": "kimi-generated",
            "author_type": "external-agent-generated",
            "author_model": "kimi-code",
            "language": "en",
            "license": "project-local",
            "source_ref": "fake",
        },
    }


def sft(index, seed):
    return {
        "messages": [
            {"role": "user", "content": "Summarize the idea briefly."},
            {"role": "assistant", "content": f"<action>\n<reasoning>Answer directly.</reasoning>\n<tool>agent.finish</tool>\n<content>This is a concise synthetic answer for fixture {seed}-{index}.</content>\n</action>"},
        ],
        "tags": ["kimi_synthetic", "language:en"],
        "meta": {
            "id": f"fake-sft-{seed}-{index:04d}",
            "split": "train",
            "provenance": "kimi-generated",
            "author_type": "external-agent-generated",
            "author_model": "kimi-code",
            "quality_tier": "high",
            "domain": "synthetic-sft",
            "skill": "summarization",
            "toolset": "none",
            "language": "en",
            "safety_scope": "workspace-safe",
            "license": "project-local",
            "source_ref": "fake",
            "mode": "sft",
            "prompt_version": "v1",
        },
    }


if __name__ == "__main__":
    main()
