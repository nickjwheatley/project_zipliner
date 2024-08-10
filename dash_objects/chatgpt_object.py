from openai import OpenAI
from config import config

def get_chatgpt_response(zip_code, config_filepath='SECRETS.ini'):
    """
    Get a summary on a specific zip code from chatGPT
    :param zip_code: string indicating desired zip code
    :param api_key: string OpenAI API key
    :return: string response from chatGPT
    """
    api_key = config(config_filepath, 'chatgpt')['key']

    client=OpenAI(api_key=api_key)

    user_pref = 'Crime rate, home valuation, job market, commute time, public and private school quality and ranking, and affordability.'

    prompt = f"Provide a compelling argument for why the home buyer should consider {zip_code} based on the following preferences {user_pref}. Don't embellish the response if a preference is not positively reflected in the zip code. For example, if the zip code has a relatively high crime rate, state that it does have a high crime rate. Please keep the response 100 words or less. Just get right to the point in the response, no need to restate purpose"

    completion = client.chat.completions.create(
      model="gpt-4o-2024-05-13",
      messages=[
        {"role": "system", "content": prompt},
        {"role": "user", "content": ""}
      ]
    )

    zip_feedback = completion.choices[0].message.content
    return zip_feedback