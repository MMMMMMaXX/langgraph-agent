from app.constants.tags import CITY_TAGS, TAG_EXTRACTION_TERMS


def extract_tags(text: str) -> list[str]:
    tags = []
    normalized_text = (text or "").lower()

    for tag, terms in TAG_EXTRACTION_TERMS.items():
        for term in terms:
            if term in text or term.lower() in normalized_text:
                tags.append(tag)
                break

    for city in CITY_TAGS:
        if city in text:
            tags.append(city)

    return tags
