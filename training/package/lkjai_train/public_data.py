from .corpus_source import tagged_contents, texts


def topic_tuple(item: dict) -> tuple[str, str, str]:
    return (item["topic"], item["practice"], item["failure"])


GENERAL_TOPICS = [topic_tuple(item) for item in tagged_contents("general", "general_topic")]
AUDIENCES = texts("general", "audience")
DELIVERABLES = texts("general", "deliverable")
CONSTRAINTS = texts("general", "constraint")
ANGLES = texts("general", "angle")
SAFETY_BOUNDARIES = texts("general", "safety_boundary")
SAFETY_REQUESTS = [item["template"] for item in tagged_contents("general", "safety_request")]
SAFER_ALTERNATIVES = texts("general", "safer_alternative")
PUBLIC_SOURCE_METADATA = tagged_contents("general", "public_source_metadata")
