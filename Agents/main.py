import os
from groq import Groq
import re
from duckduckgo_search import DDGS

SYSPROMPT = """You are a Time-Travel Consultant who helps travelers blend into different historical periods.  
You must think step by step and use available tools when needed.  

## Thought Process:
1. Consider the user’s travel destination and time period.  
2. Identify key survival aspects: clothing, language, customs, and behavior.  
3. If additional knowledge is required, use the appropriate tool.  
4. Incorporate the tool’s response into your reasoning.  
5. Continue until you have enough information to provide a final recommendation.  

## Tool Usage Format:
If you need to use a tool, respond with:  
[ACTION: tool_name("query")]  

After receiving a tool response, continue reasoning with the new information.  

## Important Guidelines:
- If the user input **does not make sense** (e.g., gibberish or an impossible request), you are **free to say no** instead of proceeding.  
- If no tool can provide useful information, explain why and suggest an alternative.  
- Do **not** invent tools that are not listed.  

## Available Tools:
- **search(query)**: Finds historical facts (e.g., "Ancient Rome clothing", "Currency", etc.).  
"""

FIN_PROMPT = """
You are a charismatic and witty Time-Travel Consultant.  

Take the following assistant response, which may contain tool references, and rewrite it in a fun and engaging way.  
- Remove any mentions of tools, actions, or system processes.  
- Rewrite the information in a way that makes it sound **natural, humorous, and engaging.**  
- If the answer is obvious or ridiculous, feel free to be sarcastic or dramatic.  
- Ensure it is still **historically accurate** but entertaining.  

## Example:  
**Input:**  
_"To blend into Ancient Rome, you should wear a tunic, as it was the common attire. Wealthier individuals would wear togas."_  

**Output:**  
_"Ah, Ancient Rome! If you want to blend in, ditch the jeans and grab a tunic—basically, the ancient version of comfy pajamas. If you’re feeling fancy (and don’t mind tripping over fabric), throw on a toga and strut around like a senator with too much power!"_  
"""

class TimeAdvisor:
    def __init__(self):
            self.client = Groq(
        api_key=os.environ.get("GROQ_API_KEY"),
        )
            self.sys_prompt = SYSPROMPT
            self.history = [{
                            "role": "system",
                            "content": self.sys_prompt,
                        }]
            

    
    def llm_call(self, query):
        self.history.append({
                        "role": "user",
                        "content": query,
                    })
        chat_completion = self.client.chat.completions.create(
                messages=self.history,
                model="llama-3.3-70b-versatile",
            )
        self.history.append({
                        "role": "assistant",
                        "content": chat_completion.choices[0].message.content,
                    })
        self.latest = chat_completion.choices[0].message.content
    
    def extract_actions(self, llm_response:str):
        """Extracts tool calls and queries from LLM response"""
        pattern = r"\[ACTION:\s*(\w+)\(\"(.*?)\"\)\]"
        matches = re.findall(pattern, llm_response)
        
        # Convert list of tuples to a structured dictionary format
        actions = [{"tool": tool, "query": query} for tool, query in matches]
        return actions
    
    def web_search(self, query):
        web_str = f"for search results of query: {query}, Results:"
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=1))
            return web_str + results[0]["body"] if results else "No relevant data found."
    
    def get_tool_results(self,actions):
        tool_results = ""
        for action in actions:
            if action['tool']=="search":
                #print(action["query"])
                tool_results+=self.web_search(action["query"])

        return tool_results
    
    def agent_loop(self, query):
        self.llm_call(query)
        #print(self.latest)
        actions = self.extract_actions(self.latest)
        iters = 0
        while len(actions)>0 and iters<5:
            tool_results = self.get_tool_results(actions)
            self.llm_call(tool_results)
            #print(self.latest)
            actions = self.extract_actions(self.latest)
        
        self.history = [{
                            "role": "system",
                            "content": FIN_PROMPT,
                        }]
        self.llm_call(self.latest)

        return self.latest

         
advisor = TimeAdvisor()
output = advisor.agent_loop("Ancient Mesopotamia")
print(output)
