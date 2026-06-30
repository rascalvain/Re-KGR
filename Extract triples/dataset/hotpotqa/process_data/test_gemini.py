from google import genai

client = genai.Client(api_key="sk-IQ8vi7XzSOgnTAW805DchQy2YVSOA8q6WYb7vUZRYOHKN6vN",http_options={"base_url":"https://api.openai-proxy.org/google"})

response = client.models.generate_content(
    model="gemini-2.5-flash-lite", contents="Explain how AI works in a few words"
)
print(response.text)