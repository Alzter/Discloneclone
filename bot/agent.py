import pandas as pd
from typing import Literal
import sys
import parser
sys.path.append("../src/")
from utils import LocalPLM, LocalModelArguments
from dataclasses import dataclass, field

@dataclass
class ChatUnderstanding():
    topic : str = field(
        metadata={"help": "Overview of what Alzterbot is talking about with the other users."}
    )
    relationship : str = field(
        metadata={"help": "Brief description of AlzterBot's relationship with the other users."}
    )
    interest : str = field(
        metadata={"help": "How interested AlzterBot is feeling about the conversation."}
    )

class Agent:
    def __init__(self):
        args = LocalModelArguments(
            model_name_or_path = "microsoft/Phi-4-mini-instruct",
            cuda_devices = "0",
            use_4bit_quantization = True,
            bnb_4bit_quant_type = "nf4",
            bnb_4bit_compute_dtype = "float16",
            use_nested_quant = True,
            use_reentrant = True
        )

        self.model = LocalPLM(args)

    def gen_prompt(self, messages : pd.DataFrame, prompt : str, context : str | None = None, context_role : Literal["system", "user"] = "system") -> list[dict[str,str]]:
        """
        Generate a Chat Template prompt to perform NLP tasks on a Discord conversation history.

        Args:
            messages (DataFrame): The conversation history.
            prompt (str): The system prompt to give the LLM.
            context (str, optional): Optional additional information related to the conversation. If provided, aids LLM performance.
            context_role (Literal["system", "user"]) : Whether to append the context to the system prompt or the conversation history. Adding context to the system prompt usually yields better results. Defaults to "system".
        
        Returns:
            prompt (list[dict[str,str]]): The prompt in chat templates.
        """
        messages_str = parser.to_string(messages, context= context if context_role == "user" else None)
        
        if context_role == "system" and context: prompt += f"\nContext: {context}.\nAnswer concisely."

        prompt_template = [{"role":"system","content":prompt}]

        prompt_template.append({"role":"user","content":messages_str})

        return prompt_template

    def understand_conversation(self, messages : pd.DataFrame, target_user : str, context : str | None = None, context_role : Literal["system", "user"] = "system", tokens:int=128) -> dict:
        """
        Use NLP to understand the meaning of a Discord conversation history from a third-person perspective.

        Returns three analyses of the conversation:
            - Topic: The topic of the conversation between the users.
            - Relationship: The relationship between the target user and other users.
            - Interest: The level of interest from the target user in the conversation.

        The analysis is done sequentially, from back to front:
            1. The personal interest of the target user in the conversation is gauged,
            2. The level of interest is used to assess the relationship between the users,
            3. The users' relationship is used as context when interpreting the subject of their conversation.

        Args:
            messages (DataFrame): The conversation history.
            target_user (str): Which user to focus on when analysing the conversation.
            context (str, optional): Optional additional information related to the conversation. If provided, aids LLM performance.
            context_role (Literal["system", "user"]) : Whether to append the context to the system prompt or the conversation history. Adding context to the system prompt usually yields better results. Defaults to "system".

        Returns:
            understanding (dict): The analysis of the conversation.
        """
        interest_prompt=f"Read the following conversation and tell me how interested {target_user} sounds in it. Be succinct."
        interest_prompt = self.gen_prompt(messages, interest_prompt, context=context, context_role=context_role)
        interest = self.model.generate(interest_prompt,max_new_tokens = 64).text

        relationship_prompt= f"Read the following conversation history and tell me what you think the relationship is between the users. Answer succinctly."

        if context: context += ", " + interest
        else: context = interest
        relationship_prompt = self.gen_prompt(messages, relationship_prompt, context=context, context_role=context_role)
        relationship = self.model.generate(relationship_prompt,temperature=1,max_new_tokens = 128).text

        topic_prompt="Read the following conversation history and tell me what was discussed. Answer succinctly."
        topic_prompt = self.gen_prompt(messages, topic_prompt, context=relationship + ", " + interest, context_role=context_role)
        topic = self.model.generate(topic_prompt,temperature=1,max_new_tokens = tokens).text

        #return f"Conversation topic:\n{topic}\n\nRelationship between users:\n{relationship}\n\nPersonal interest:\n{interest}"
        return ChatUnderstanding(interest=interest,relationship=relationship,topic=topic)

        return {"interest":interest,"relationship":relationship,"topic":topic}

    def understand_conversation_pov(self, messages : pd.DataFrame, target_user : str, context : str | None = None, context_role : Literal["system", "user"] = "system", tokens:int=128) -> dict:
        """
        Use NLP to understand the meaning of a Discord conversation history from the perspective of a given user in first-person.

        Returns three analyses of the conversation:
            - Topic: The topic of the conversation between the users.
            - Relationship: The relationship between the target user and other users.
            - Interest: The level of interest from the target user in the conversation.

        The analysis is done sequentially, from back to front:
            1. The personal interest of the target user in the conversation is gauged,
            2. The level of interest is used to assess the relationship between the users,
            3. The users' relationship is used as context when interpreting the subject of their conversation.

        Args:
            messages (DataFrame): The conversation history.
            target_user (str): Which user to focus on when analysing the conversation.
            context (str, optional): Optional additional information related to the conversation. If provided, aids LLM performance.
            context_role (Literal["system", "user"]) : Whether to append the context to the system prompt or the conversation history. Adding context to the system prompt usually yields better results. Defaults to "system".

        Returns:
            understanding (dict): The analysis of the conversation.
        """
        # Get a string for the name of all other users
        other_users = " and ".join([i for i in messages.Author.unique() if not i == target_user])

        interest_prompt=f"Your name is {target_user}. Read one of your past text conversations with {other_users} and tell me how interested you were during it. Respond with first person perspective. Be succinct."
        interest_prompt = self.gen_prompt(messages, interest_prompt, context=context, context_role=context_role)
        interest = self.model.generate(interest_prompt,max_new_tokens = 64).text

        relationship_prompt= f"Your name is {target_user}. Read one of your past text conversations with {other_users} and tell me what your relationship is with them. Respond with first person perspective. Be succinct."

        if context: context += ", " + interest
        else: context = interest
        relationship_prompt = self.gen_prompt(messages, relationship_prompt, context=context, context_role=context_role)
        relationship = self.model.generate(relationship_prompt,temperature=1,max_new_tokens = 128).text

        topic_prompt=f"Your name is {target_user}. Read one of your past text conversations with {other_users} and tell me what you were talking about. Respond with first person perspective. Be succinct."
        topic_prompt = self.gen_prompt(messages, topic_prompt, context=relationship + ", " + interest, context_role=context_role)
        topic = self.model.generate(topic_prompt,temperature=1,max_new_tokens = tokens).text

        #return f"Conversation topic:\n{topic}\n\nMy relationship with {other_users}:\n{relationship}\n\nMy interest in the conversation:\n{interest}"
        # return {"interest":interest,"relationship":relationship,"topic":topic}
        return ChatUnderstanding(interest=interest,relationship=relationship,topic=topic)

    def reply(self, messages : pd.DataFrame, context : str | None = None) -> str:
        """
        Reply to a Discord conversation using a DataFrame of conversation history.
        """
        prompt = self.gen_prompt(messages,
                              prompt="You are alzter. Read this conversation history and write a message to continue the conversation.",
                              context=context)

        response = self.model.generate(prompt, max_new_tokens=128).text

        return response
