import json
import inspect
from openai import OpenAI
from IPython.display import display, HTML
import markdown

class Tools:
    def __init__(self):
        self.tools_list = []
        self.functions = {}

    def add_tool(self, function, description):
        tool_schema = {
            "type": "function",
            "function": {
                "name": function.__name__,
                "description": description,
                "parameters": self._get_function_parameters(function)
            }
        }
        self.tools_list.append(tool_schema)
        self.functions[function.__name__] = function
    
    def _get_function_parameters(self, function):
        """Extract function parameters and create OpenAI-compatible schema"""
        sig = inspect.signature(function)
        properties = {}
        required = []
        
        for param_name, param in sig.parameters.items():
            param_info = {"type": "string"} 
         
            if param.default == inspect.Parameter.empty:
                required.append(param_name)
            
            properties[param_name] = param_info
        
        return {
            "type": "object",
            "properties": properties,
            "required": required
        }

    def get_tools(self):
        return self.tools_list

    def execute_tool(self, tool_call):
        function_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)
        function = self.functions[function_name]
        result = function(**arguments)
        
        return {
            "tool_call_id": tool_call.id,
            "role": "tool",
            "name": function_name,
            "content": str(result),
        }


class ChatInterface:
    def input(self):
        return input("You: ")
    
    def display(self, message):
        print(message)

    def display_function_call(self, tool_call, result):
        call_html = f"""
            <details>
            <summary>Function call: {tool_call.function.name}()</summary>
            <div>
                <b>Arguments:</b>
                <pre>{json.dumps(json.loads(tool_call.function.arguments), indent=2)}</pre>
            </div>
            <div>
                <b>Result:</b>
                <pre>{result['content']}</pre>
            </div>
            </details>
        """
        display(HTML(call_html))

    def display_response(self, message):
        response_html = markdown.markdown(message.content)
        html = f"""
            <div style='margin: 10px 0; padding: 10px; border-left: 3px solid #ddd;'>
                <div style='font-weight: bold;'>Assistant:</div>
                <div>{response_html}</div>
            </div>
        """
        display(HTML(html))


class ChatAssistant:
    def __init__(self, tools, system_prompt, chat_interface, client):
        self.tools = tools
        self.system_prompt = system_prompt
        self.chat_interface = chat_interface
        self.client = client
    
    def gpt(self, chat_messages):
        return self.client.chat.completions.create(
            model="deepseek-chat",
            messages=chat_messages,
            tools=self.tools.get_tools(),
            tool_choice="auto"
        )

    def run(self):
        chat_messages = [{"role": "system", "content": self.system_prompt}]

        while True:
            try:
                # Get user input
                question = self.chat_interface.input()
                if question.lower() == "stop":
                    self.chat_interface.display("Chat ended.")
                    break

                # Add user message to history
                chat_messages.append({"role": "user", "content": question})

                # Conversation loop (may involve multiple tool calls)
                while True:
                    # Get model response
                    response = self.gpt(chat_messages)
                    assistant_message = response.choices[0].message
                    
                    # Add assistant message to history
                    assistant_message_dict = {
                        "role": "assistant",
                        "content": assistant_message.content or None
                    }
                    if assistant_message.tool_calls:
                        assistant_message_dict["tool_calls"] = assistant_message.tool_calls
                    chat_messages.append(assistant_message_dict)

                    # Handle tool calls if any
                    if assistant_message.tool_calls:
                        for tool_call in assistant_message.tool_calls:
                            tool_result = self.tools.execute_tool(tool_call)
                            chat_messages.append(tool_result)
                            self.chat_interface.display_function_call(tool_call, tool_result)
                    else:
                        # No tool calls - display final response
                        if assistant_message.content:
                            self.chat_interface.display_response(assistant_message)
                        break

            except Exception as e:
                self.chat_interface.display(f"Error: {str(e)}")
                break
