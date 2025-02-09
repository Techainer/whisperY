import google.generativeai as genai

genai.configure(api_key="")

def upload_to_gemini(path, mime_type=None):
  """Uploads the given file to Gemini.
  """
  file = genai.upload_file(path, mime_type=mime_type)
  print(f"Uploaded file '{file.display_name}' as: {file.uri}")
  return file

def delete_from_gemini(file):
  """Uploads the given file to Gemini.
  """
  print(f"Deleted file '{file.display_name}' as: {file.uri}")
  genai.delete_file(file)

# Create the model
generation_config = {
  "temperature": 0,
  "top_p": 0.95,
  "top_k": 40,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
  model_name="gemini-2.0-flash-exp",
  generation_config=generation_config,
)

file = upload_to_gemini("untitled.wav", mime_type="audio/wav")

contents = [file, "transcribe this audio, return result in vietnamese"]

response = model.generate_content(contents)

print(response.text)

delete_from_gemini(file)