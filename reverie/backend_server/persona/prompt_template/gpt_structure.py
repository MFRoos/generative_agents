"""
Author: Joon Sung Park (joonspk@stanford.edu)

File: gpt_structure.py
Description: Wrapper functions for calling Ollama via OpenAI-compatible API.
"""
import json
import random
import time

from openai import OpenAI

from utils import *

OLLAMA_MODEL = "gemma3:4b"
OLLAMA_EMBED_MODEL = "nomic-embed-text"
OLLAMA_BASE_URL = "http://localhost:11434/v1"

client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")


def temp_sleep(seconds=0.1):
  time.sleep(seconds)


def _chat_request(prompt, model=OLLAMA_MODEL, temperature=1.0, max_tokens=None):
  """Core chat completion call to Ollama."""
  kwargs = dict(
    model=model,
    messages=[{"role": "user", "content": prompt}],
    temperature=temperature,
  )
  if max_tokens:
    kwargs["max_tokens"] = max_tokens
  response = client.chat.completions.create(**kwargs)
  return response.choices[0].message.content


def ChatGPT_single_request(prompt):
  temp_sleep()
  return _chat_request(prompt)


# ============================================================================
# #####################[SECTION 1: CHAT STRUCTURE] ############################
# ============================================================================

def GPT4_request(prompt):
  """Forwards to the configured Ollama model (originally GPT-4)."""
  temp_sleep()
  try:
    return _chat_request(prompt)
  except Exception as e:
    print(f"Ollama ERROR: {e}")
    return "Ollama ERROR"


def ChatGPT_request(prompt):
  """Forwards to the configured Ollama model (originally ChatGPT)."""
  try:
    return _chat_request(prompt)
  except Exception as e:
    print(f"Ollama ERROR: {e}")
    return "Ollama ERROR"


def GPT4_safe_generate_response(prompt,
                                example_output,
                                special_instruction,
                                repeat=3,
                                fail_safe_response="error",
                                func_validate=None,
                                func_clean_up=None,
                                verbose=False):
  prompt = 'GPT-3 Prompt:\n"""\n' + prompt + '\n"""\n'
  prompt += f"Output the response to the prompt above in json. {special_instruction}\n"
  prompt += "Example output json:\n"
  prompt += '{"output": "' + str(example_output) + '"}'

  if verbose:
    print("OLLAMA PROMPT")
    print(prompt)

  for i in range(repeat):
    try:
      curr_gpt_response = GPT4_request(prompt).strip()
      end_index = curr_gpt_response.rfind('}') + 1
      curr_gpt_response = curr_gpt_response[:end_index]
      curr_gpt_response = json.loads(curr_gpt_response)["output"]

      if func_validate(curr_gpt_response, prompt=prompt):
        return func_clean_up(curr_gpt_response, prompt=prompt)

      if verbose:
        print("---- repeat count: \n", i, curr_gpt_response)
        print("~~~~")
    except:
      pass

  return False


def ChatGPT_safe_generate_response(prompt,
                                   example_output,
                                   special_instruction,
                                   repeat=3,
                                   fail_safe_response="error",
                                   func_validate=None,
                                   func_clean_up=None,
                                   verbose=False):
  prompt = '"""\n' + prompt + '\n"""\n'
  prompt += f"Output the response to the prompt above in json. {special_instruction}\n"
  prompt += "Example output json:\n"
  prompt += '{"output": "' + str(example_output) + '"}'

  if verbose:
    print("OLLAMA PROMPT")
    print(prompt)

  for i in range(repeat):
    try:
      curr_gpt_response = ChatGPT_request(prompt).strip()
      end_index = curr_gpt_response.rfind('}') + 1
      curr_gpt_response = curr_gpt_response[:end_index]
      curr_gpt_response = json.loads(curr_gpt_response)["output"]

      if func_validate(curr_gpt_response, prompt=prompt):
        return func_clean_up(curr_gpt_response, prompt=prompt)

      if verbose:
        print("---- repeat count: \n", i, curr_gpt_response)
        print("~~~~")
    except:
      pass

  return False


def ChatGPT_safe_generate_response_OLD(prompt,
                                       repeat=3,
                                       fail_safe_response="error",
                                       func_validate=None,
                                       func_clean_up=None,
                                       verbose=False):
  if verbose:
    print("OLLAMA PROMPT")
    print(prompt)

  for i in range(repeat):
    try:
      curr_gpt_response = ChatGPT_request(prompt).strip()
      if func_validate(curr_gpt_response, prompt=prompt):
        return func_clean_up(curr_gpt_response, prompt=prompt)
      if verbose:
        print(f"---- repeat count: {i}")
        print(curr_gpt_response)
        print("~~~~")
    except:
      pass
  print("FAIL SAFE TRIGGERED")
  return fail_safe_response


# ============================================================================
# ###################[SECTION 2: COMPLETION-STYLE REQUESTS] ###################
# ============================================================================

def GPT_request(prompt, gpt_parameter):
  """
  Originally used openai.Completion; now routes through chat completions
  since Ollama only supports the chat endpoint.
  """
  temp_sleep()
  try:
    temperature = gpt_parameter.get("temperature", 1.0)
    max_tokens = gpt_parameter.get("max_tokens", None)
    stop = gpt_parameter.get("stop", None)

    kwargs = dict(
      model=OLLAMA_MODEL,
      messages=[{"role": "user", "content": prompt}],
      temperature=temperature,
    )
    if max_tokens:
      kwargs["max_tokens"] = max_tokens
    if stop:
      kwargs["stop"] = stop

    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content
  except Exception as e:
    print(f"TOKEN LIMIT EXCEEDED or Ollama ERROR: {e}")
    return "TOKEN LIMIT EXCEEDED"


def generate_prompt(curr_input, prompt_lib_file):
  """
  Takes in the current input and the path to a prompt file. Replaces
  !<INPUT N>! placeholders with the actual inputs.
  """
  if type(curr_input) == type("string"):
    curr_input = [curr_input]
  curr_input = [str(i) for i in curr_input]

  f = open(prompt_lib_file, "r")
  prompt = f.read()
  f.close()
  for count, i in enumerate(curr_input):
    prompt = prompt.replace(f"!<INPUT {count}>!", i)
  if "<commentblockmarker>###</commentblockmarker>" in prompt:
    prompt = prompt.split("<commentblockmarker>###</commentblockmarker>")[1]
  return prompt.strip()


def safe_generate_response(prompt,
                           gpt_parameter,
                           repeat=5,
                           fail_safe_response="error",
                           func_validate=None,
                           func_clean_up=None,
                           verbose=False):
  if verbose:
    print(prompt)

  for i in range(repeat):
    curr_gpt_response = GPT_request(prompt, gpt_parameter)
    if func_validate(curr_gpt_response, prompt=prompt):
      return func_clean_up(curr_gpt_response, prompt=prompt)
    if verbose:
      print("---- repeat count: ", i, curr_gpt_response)
      print("~~~~")
  return fail_safe_response


def get_embedding(text, model=OLLAMA_EMBED_MODEL):
  text = text.replace("\n", " ")
  if not text:
    text = "this is blank"
  response = client.embeddings.create(input=[text], model=model)
  return response.data[0].embedding


if __name__ == '__main__':
  gpt_parameter = {"engine": OLLAMA_MODEL, "max_tokens": 50,
                   "temperature": 0, "top_p": 1, "stream": False,
                   "frequency_penalty": 0, "presence_penalty": 0,
                   "stop": ['"']}
  curr_input = ["driving to a friend's house"]
  prompt_lib_file = "prompt_template/test_prompt_July5.txt"
  prompt = generate_prompt(curr_input, prompt_lib_file)

  def __func_validate(gpt_response, prompt=""):
    if len(gpt_response.strip()) <= 1:
      return False
    if len(gpt_response.strip().split(" ")) > 1:
      return False
    return True

  def __func_clean_up(gpt_response, prompt=""):
    return gpt_response.strip()

  output = safe_generate_response(prompt,
                                  gpt_parameter,
                                  5,
                                  "rest",
                                  __func_validate,
                                  __func_clean_up,
                                  True)
  print(output)
