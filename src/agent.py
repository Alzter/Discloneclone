import pandas as pd
from typing import Literal
import parser
from utils.utils import LocalPLM, LocalModelArguments
from dataclasses import dataclass, field
from tqdm import tqdm

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

    def understand_conversation(self, messages : pd.DataFrame, target_user : str, context : str | None = None, context_role : Literal["system", "user"] = "system", tokens:int=128) -> ChatUnderstanding:
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

    def understand_conversation_pov(self, messages : pd.DataFrame, target_user : str, context : str | None = None, context_role : Literal["system", "user"] = "system", tokens:int=128) -> ChatUnderstanding:
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

    def predict_thought(self, conversation : pd.DataFrame, message_id : int, context="context", tokens : int = 128):
        conversation = conversation.reset_index(drop=True)[:message_id + 1] 
        target_message = conversation.iloc[message_id]
        target_text = target_message.Content
        target_user = target_message.Author
        other_users = " and ".join([i for i in conversation.Author.unique() if not i == target_user])
        
        context = self.understand_conversation_pov(conversation, target_user)
        
        thought_prompt = f"""
Your name is {target_user}. You are in a text conversation with {other_users}.
Read the conversation, then tell me what you are thinking as you say:
'{target_text}'. Answer in first-person tense. Be succinct.""".strip()
        
        thought_prompt = self.gen_prompt(conversation, thought_prompt, context=context)
        
        predicted_thought = self.model.generate(thought_prompt, temperature=1, max_new_tokens=tokens).text
        
        return predicted_thought

    def conversation_to_dataset(self, conversation : pd.DataFrame, target_user : str, batch_size : int = 10, use_context : bool = True, initial_context : str | None = None, thinking_tokens : int = 0) -> pd.DataFrame:
        """
        Convert a Discord conversation into a supervised chat dataset from the perspective of a given user.
        This can be used to predict messages from a given user (i.e., training a model to impersonate you).
        
        Conversations are split up into smaller batches to reduce the size of each input text.
        
        Optionally, you can allow an LLM to guess what the target user was thinking for each message.
        This feature is aimed to improve LLM response precision by getting in the head of the target user.
        
        Args:
            conversation (DataFrame): The conversation to convert.
            target_user (str): Which user we're trying to predict the messages of.
            batch_size (int, optional): Maximum number of new input messages per sample. Defaults to 10.
            use_context (bool, optional): If enabled, at the start of each new batch, a summarisation of the previous batch's conversation is given as context.
                                          This helps eliminate loss of semantic meaning when slicing conversations into chunks of arbitrary size. Defaults to True.
            initial_context (str, optional): If given, provides the context for the users' relationship.
            thinking_tokens (int, optional): If > 0, predicts the thoughts of the target user for each message using a given number of tokens. Defaults to 0.
        
        Returns:
            dataset (DataFrame): NLP dataset with columns "content" (input) and "label" (output).
        """
        data = {"content" : [], "label" : []}
        
        # Get the index of each message
        indices = list(conversation.index)
        
        # Slice indices into batches / chunks
        chunks = [indices[i:i + batch_size] for i in range(0, len(indices), batch_size)]
        
        # If we want to predict the target user's thoughts
        # for each message, we should first gauge what the
        # relationship between the users is like for all
        # messages in the conversation. We can then use
        # this relationship info as context for the
        # thought prediction prompt to improve its accuracy.
        full_context = "" 
        if thinking_tokens > 0:
            if initial_context: full_context = None 
            else: full_context = self.understand_conversation_pov( conversation, target_user=target_user, context=initial_context ).relationship 
        
        # Create an empty context for now
        context_str = initial_context
        
        # For each batch
        for i, indices in enumerate(chunks):
            start_index = indices[0]
            
            # Get the indices of the target user's messages
            user_indices = conversation.iloc[indices]
            user_indices = user_indices[user_indices.Author == target_user].index
            user_indices = list(user_indices)

            other_user_indices = conversation.iloc[indices]
            other_user_indices = other_user_indices[other_user_indices.Author != target_user].index
            other_user_indices = list(user_indices)
            
            # Exclude all messages by the target user which don't come before another user
            user_indices = [i for i in user_indices if i > min(other_user_indices)]
            
            # For each user message
            for index in user_indices:
                # Get all messages which preceded it in the batch as a string
                s = start_index
                if start_index == index: s -= 1
                if s < 0: continue
                inputs = conversation.iloc[s:index]
                inputs = parser.to_string(inputs, header=True, context=context_str, target_user=target_user)
                 
                # Get the user message itself as a string
                output = conversation.iloc[index:index+1]
                output = parser.to_string(output, header=False, target_user=target_user)
                
                # Get the user's thought for the given message
                if thinking_tokens > 0:
                    
                    # We have to get the index of the user's message relative to
                    # the start of the batch for .iloc[] to work inside the batch
                    local_index = index - start_index
                    
                    # Create a context for the user's thought for the message
                    # using their relationship with the other users + conversation history
                    if full_context and not context_str: thinking_context = full_context
                    if context_str and not full_context: thinking_context = context_str
                    if full_context and context_str:
                        thinking_context = full_context + "\n" + context_str
                    
                    # Predict the user's thought for the message
                    thought = self.predict_thought( conversation.iloc[indices], local_index, context=thinking_context, tokens=thinking_tokens)
                    
                    # Enclose the thought in <thinking> tags
                    output = f"<thinking>{thought}</thinking>\n\n" + output
                
                data['content'].append(inputs)
                data['label'].append(output)
                
                # print("\n----\nIN:")
                # print(inputs)
                # print("\n----\nOUT:")
                # print(output)
                # print("----")
            
            # At the end of each batch, summarise what was discussed
            # to use as the context string for the next batch.
            # (Only do this if there are more chunks remaining)
            if i < len(chunks) - 1 and use_context:
                context_str = self.understand_conversation_pov( conversation.iloc[indices], context=context_str, target_user=target_user ).topic
            
        return pd.DataFrame(data)
    
    def conversations_to_dataset(self, conversations : list[pd.DataFrame], target_user : str, batch_size : int = 10, initial_context : str | None = None, use_context : bool = True, thinking_tokens : int = 0):
        """
        Parse all conversations from a user into a supervised text dataset.

        Args:
            conversations (list[DataFrame]): The conversations to convert.
            target_user (str): Which user we're trying to predict the messages of.
            batch_size (int, optional): Maximum number of new input messages per sample. Defaults to 10.
            use_context (bool, optional): If enabled, uses LLM generation to add synthetic context to the conversation. Defaults to True.
            initial_context (str, optional): If given, provides the context for the users' relationship.
            thinking_tokens (int, optional): If > 0, predicts the thoughts of the target user for each message using a given number of tokens. Defaults to 0.
        
        Returns:
            dataset (DataFrame): NLP dataset with columns "content" (input) and "label" (output).
        """
        data = []
        
        context_str = initial_context

        for i, conversation in enumerate(tqdm(conversations)):
            
            items = self.conversation_to_dataset(conversation,target_user, batch_size=batch_size,initial_context=context_str,use_context=use_context,thinking_tokens=thinking_tokens)
            data.append(items)
            
            if i < len(conversations) - 1 and use_context:
                context_str = self.understand_conversation_pov(conversation, target_user=target_user, context=context_str).relationship
        
        data = pd.concat(data)
        return data

    def reply(self, messages : pd.DataFrame, context : str | None = None) -> str:
        """
        Reply to a Discord conversation using a DataFrame of conversation history.
        """
        prompt = self.gen_prompt(messages,
                              prompt="You are alzter. Read this conversation history and write a message to continue the conversation.",
                              context=context)

        response = self.model.generate(prompt, max_new_tokens=128).text

        return response
