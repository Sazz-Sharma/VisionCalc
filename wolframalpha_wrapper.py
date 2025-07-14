import requests
import xml.etree.ElementTree as ET
import urllib.parse

class WolframAlphaWrapper:
    def __init__(self, appid, use_simple=True):
        self.appid = appid
        self.use_simple = use_simple  # If True, use /v1/result for fast response

    def query(self, question, detailed=False):
        if self.use_simple and not detailed:
            return self._simple_query(question)
        else:
            return self._detailed_query(question)

    def _simple_query(self, question):
        encoded_q = urllib.parse.quote_plus(question)
        url = f"https://api.wolframalpha.com/v1/result?appid={self.appid}&i={encoded_q}"

        response = requests.get(url)

        if response.status_code == 200:
            return response.text
        elif response.status_code == 501:
            return "No short answer available."
        else:
            return f"Error: {response.status_code} - {response.text}"

    def _detailed_query(self, question):
        encoded_q = urllib.parse.quote_plus(question)
        url = f"https://api.wolframalpha.com/v2/query?appid={self.appid}&input={encoded_q}&output=XML"

        response = requests.get(url)
        if response.status_code != 200:
            return f"Error: {response.status_code} - {response.text}"

        try:
            root = ET.fromstring(response.content)
            if root.attrib.get("success") != "true":
                return "Wolfram couldn't understand the input."

            result = []
            for pod in root.findall(".//pod"):
                title = pod.attrib.get("title", "No Title")
                subpods = pod.findall(".//subpod")
                contents = [subpod.findtext("plaintext", default="") for subpod in subpods]
                contents = "\n".join(filter(None, contents)).strip()
                if contents:
                    result.append(f"{title}:\n{contents}")
            return "\n\n".join(result) if result else "No interpretable output found."

        except ET.ParseError:
            return "Failed to parse XML response."
    

