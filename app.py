import asyncio
import json
import gradio as gr
from openai import AsyncOpenAI, OpenAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Configuration
XAI_API_KEY = os.getenv("XAI_API_KEY")
client = AsyncOpenAI(
    api_key=XAI_API_KEY,
    base_url="https://api.x.ai/v1",
)

simple_client = OpenAI(
    api_key=XAI_API_KEY,
    base_url="https://api.x.ai/v1",
)

# Load agent personalities
with open('data/agent_bank.json', 'r') as f:
    AGENT_BANK = json.load(f)['agents']

class MultiAgentConversationalSystem:
    def __init__(self, api_client):
        self.client = api_client
        self.agents = AGENT_BANK
        self.first_stage_results = []
        self.conversation_histories = {}
        self.manager_agent = {
            "first_name": "Alex",
            "last_name": "Policymaker",
            "expertise": "Policy Strategy and Synthesis",
            "personality": "Strategic, analytical, and focused on comprehensive understanding"
        }

    async def first_stage_analysis(self, policy):
        """First stage: Agents analyze policy and provide reasoning with yes/no answer"""
        async def agent_policy_analysis(agent):
            agent_context = "\n".join([
                f"{key}: {value}" for key, value in agent.items()
            ])

            prompt = f"""
            Agent Profile:
            {agent_context}

            Policy/Topic: {policy}
            
            Task:
            1. Carefully analyze the policy/topic using ALL aspects of your defined personality and expertise.
            2. Provide a clear YES or NO answer.
            3. Explain your reasoning in 2-3 detailed paragraphs.
            4. Leverage every aspect of your defined characteristics to provide a comprehensive analysis.

            Format your response as:
            - Agent: {agent['first_name']} {agent['last_name']}
            - Answer: YES/NO
            - Reasoning: [Detailed explanation drawing from ALL your defined attributes]
            """
            
            try:
                response = await self.client.chat.completions.create(
                    model="grok-beta",
                    messages=[{"role": "user", "content": prompt}]
                )
                agent_response = {
                    "full_name": f"{agent['first_name']} {agent['last_name']}",
                    "expertise": agent['expertise'],
                    "full_agent_context": agent,
                    "full_response": response.choices[0].message.content
                }
                
                return agent_response
            except Exception as e:
                return {
                    "full_name": f"{agent['first_name']} {agent['last_name']}",
                    "full_agent_context": agent,
                    "full_response": f"Error: {str(e)}"
                }

        tasks = [agent_policy_analysis(agent) for agent in self.agents]
        self.first_stage_results = await asyncio.gather(*tasks)
        
        # {chr(10).join([f"- {result['full_name']}: {result['full_response'].split('Reasoning:')[1].strip()}" for result in self.first_stage_results])}
        
        summary_prompt = f"""
        Policy/Topic: {policy}

        Agent Analyses Summary:
        {self.first_stage_results}

        Your Task:
        1. Synthesize the diverse agent perspectives into a comprehensive policy overview.
        2. Identify key insights, potential challenges, and strategic recommendations.
        3. Provide a balanced and strategic assessment of the policy.
        """

        manager_name = f"{self.manager_agent['first_name']} {self.manager_agent['last_name']}"
        self.conversation_histories[manager_name] = [
            {"role": "system", "content": f"""
            You are {manager_name}, a strategic policy analyst with expertise in {self.manager_agent['expertise']}.
            You synthesize complex perspectives and provide strategic policy insights.

            Initial Policy Summary:
            {summary_prompt}
            """}
        ]
        
        return self.first_stage_results

    async def manager_summary(self, policy):
        try:
            response = await self.client.chat.completions.create(
                model="grok-beta",
                messages=[{"role": "user", "content": f"""Summarized this.\n\n{policy}"""}],
                stream=False
            )
            
            manager_summary = response.choices[0].message.content
            return manager_summary
        
        except Exception as e:
            return f"Summary generation error: {str(e)}"

    async def agent_conversation(self, agent_name, message, history):
        if agent_name not in self.conversation_histories:
            agent_context = next((agent for agent in self.first_stage_results 
                                  if f"{agent['full_agent_context']['first_name']} {agent['full_agent_context']['last_name']}" == agent_name), 
                                 None)
            if not agent_context:
                return "Agent not found."
            
            self.conversation_histories[agent_name] = [
                {"role": "system", "content": f"""
                You are {agent_name}, an agent with the following profile:
                Expertise: {agent_context['expertise']}
                
                Approach the conversation from your unique perspective, 
                drawing on your expertise and personality.
                """}
            ]
        
        conversation_history = self.conversation_histories[agent_name].copy()
        conversation_history.append({"role": "user", "content": message})
        
        try:
            response = await self.client.chat.completions.create(
                model="grok-beta",
                messages=conversation_history,
                stream=True
            )
            
            agent_response = response.choices[0].message.content
            self.conversation_histories[agent_name].append(
                {"role": "user", "content": message}
            )
            self.conversation_histories[agent_name].append(
                {"role": "assistant", "content": agent_response}
            )
            
            return agent_response
        
        except Exception as e:
            return f"Conversation error: {str(e)}"

# Chat
def predict(message, history, policy_summary):

    system_prompt = """\
    You are an assistant, that work as a Policymaker. Expertise in Policy Strategy and Synthesis.
    With a personality of Strategic, analytical, and focused on comprehensive understanding.
    """
    
    policy_summary_prompt = f"""\
    Here are the policy summary of professtional role in the country.
    {policy_summary}
    """
    
    history_openai_format = [{"role": "system", "content": system_prompt}]
    history_openai_format.append({"role": "user", "content": policy_summary_prompt})
    
    for human, assistant in history:
        if isinstance(human, str) and human.strip():
            history_openai_format.append({"role": "user", "content": human})
        if isinstance(assistant, str) and assistant.strip():
            history_openai_format.append({"role": "assistant", "content": assistant})
    
    history_openai_format.append({"role": "user", "content": message})
    
    print("history_openai_format:", history_openai_format)
  
    response = simple_client.chat.completions.create(
        model='grok-beta',
        messages=history_openai_format,
        temperature=0.6,
        stream=True
    )

    partial_message = ""
    for chunk in response:
        if chunk.choices[0].delta.content is not None:
            partial_message += chunk.choices[0].delta.content
            yield partial_message

def chat_bot(user_input, history, policy_summary):
    bot_response_generator = predict(user_input, history, policy_summary)
    history.append((user_input, ""))

    for bot_response in bot_response_generator:
        history[-1] = (user_input, bot_response)
        yield "", history

def create_gradio_interface():
    multi_agent_system = MultiAgentConversationalSystem(client)

    def get_manager_summary(policy):
        summary = asyncio.run(multi_agent_system.manager_summary(policy))
        return summary

    def agent_chat(agent_name, message, history, summary_policy):
        response = asyncio.run(multi_agent_system.agent_conversation(agent_name, message, history, summary_policy))
        history.append((message, response))
        return "", history

    def first_stage_process(policy):
        gr.Info("Running Agent Parallel Please Wait....")
        results = asyncio.run(multi_agent_system.first_stage_analysis(policy))
        formatted_output = "🔍 First Stage: Agent Policy Analyses\n\n"
        for result in results:
            formatted_output += f"**{result['full_name']}:**\n{result['full_response']}\n\n{'='*50}\n\n"
        gr.Info("Running Agent Done!")

        return formatted_output

    with gr.Blocks() as demo:
        gr.Markdown("# 🌐 Two-Stage Multi-Agent Policy Analysis")
        
        with gr.Tab("First Stage: Policy Analysis"):
            policy_input = gr.Textbox(label="Policy/Topic")
            first_stage_btn = gr.Button("Analyze Policy")
            policy_summary = gr.Markdown(label="Agent Perspectives")
            
            first_stage_btn.click(
                fn=first_stage_process, 
                inputs=policy_input, 
                outputs=[policy_summary]
            )
            
        with gr.Tab("Second Stage: Chat with Policy Maker"):
            chatbot = gr.Chatbot(elem_id="chatbot")
            msg = gr.Textbox(placeholder="Put your message here...")

            with gr.Row():
                clear = gr.Button("Clear History")
                send = gr.Button("Send Message", variant="primary")
                
            gr.Examples(
                examples=[
                    "Should I implement this?", 
                    "Can you recommend what should i do?", 
                ], 
                inputs=msg,
            )

            clear.click(lambda: [], [], chatbot)
            msg.submit(chat_bot, [msg, chatbot, policy_summary], [msg, chatbot])
            send.click(chat_bot, [msg, chatbot, policy_summary], [msg, chatbot])
            
    return demo

def main():
    app = create_gradio_interface()
    app.launch()

if __name__ == "__main__":
    main()
