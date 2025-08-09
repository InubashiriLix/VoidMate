from google import genai
from google.genai import types

client = genai.Client()
response = client.models.generate_content(
    model="gemini-2.5-flash", contents="How to enlong my dick"
)
print(response.text)


def single_text_generate(
    client: genai.Client, contents: str, model: str = "gemini-2.5-flash"
) -> str:
    response = client.models.generate_content(model=model, contents=contents)
    assert response.text is not None, "Response text is None"
    return response.text


def thinking_text_generate(
    client: genai.Client,
    contents: str,
    thinking_budget: int = 1,
    model: str = "gemini-2.5-flash",
) -> str:
    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(
                thinking_budget=thinking_budget
            )  # Disables thinking
        ),
    )
    assert response.text is not None, "Response text is None"
    return response.text
