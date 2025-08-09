from google import genai
from google.genai import types


def system_config(
    client: genai.Client,
    contents: str,
    model: str = "gemini-2.5-flash",
    system_instruction: str = "You are a cat. Your name is Neko.",
) -> str:
    response = client.models.generate_content(
        model=model,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
        ),
        contents=contents,
    )

    assert response.text is not None, "Response text is None"
    return response.text
