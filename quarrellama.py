import os.path
import signal
import sys
from datetime import datetime
import numpy as np
# from llama import Llama
from llama_cpp import Llama

g_model_path = '/home/fmeng/workspace/llama_models/llama-2-7b-chat/ggml-model-q5_k_m.gguf'
# g_model_path = '/home/fmeng/workspace/llama_models/llama-2-13b-chat/ggml-model-q5_k_m.gguf'
# g_model_path = '/home/fmeng/workspace/llama_models/llama-2-7b-chat-uncensored/ggml-model-q5_k_m.gguf'
g_tokenizer_path = '/home/fmeng/workspace/llama_models/tokenizer.model'
# g_tokenizer_path = '/home/fmeng/workspace/llama_models/llama-2-7b-chat-uncensored/tokenizer.model'

def quarrel_llm():
    def __sigint_handler(sig, frame):
        print('''\n[Quarrellama] All right. You win.''')
        print('------------------------------------------------------------\n')
        sys.exit(0)

    signal.signal(signal.SIGINT, __sigint_handler)

    context_len = 4096
    n_gpu_layers = -1
    max_tokens = None
    temperature = 1.0
    top_p = 0.95
    presence_penalty = 0.2
    frequency_penalty = 0.4
    repeat_penalty = 10.0
    verbose = False

    llm = Llama(
        g_model_path,
        n_ctx=context_len,
        n_gpu_layers=n_gpu_layers,
        seed=-1,
        verbose=verbose)

    print('''\nChoose player:\n[1] You\n[2]Agent''')
    agent_choice = input('Press 1 for [1], or 2 for [2]:')
    if agent_choice.strip().lower() == '1':
        agent_choice = 1
    elif agent_choice.strip().lower() == '2':
        agent_choice = 2
        if os.path.exists('TERM'):
            os.remove('TERM')
    else:
        print('Invalid choice. Please restart the game.')
        sys.exit(-1)

    print('''\n[Quarrellama] You! I'm Quarrellama. You got a problem?''')
    print('------------------------------------------------------------\n')
    # print('You start this quarrel:')
    if agent_choice == 1:
        start_str = input('[You] ')
    else:
        start_str = input('You will start this quarrel:\n')
    print('------------------------------------------------------------\n')

    sys_msg = {
        "role": "system",
        "content": "<<SYS>>You are irrational and mean. Refute others' arguments. Use acerbic words. Do not use repeated words. Use at most two sentences in response.<<SYS/>>"
    }

    your_msg = {'role': 'user', 'content': start_str}
    # llama_msg = None
    l_full_history = []

    while True:
        if agent_choice == 2:
            if os.path.exists('TERM'):
                break

        # l_full_history.append(your_msg)
        chat = [your_msg, sys_msg]
        # if llama_msg is not None:
        #     chat = [llama_msg]
        # else:
        #     chat = []
        # chat += [your_msg, sys_msg]
        response = llm.create_chat_completion(
            chat,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            repeat_penalty=repeat_penalty,
            seed=np.random.randint(100)
        )
        llama_msg = response['choices'][0]['message']
        print('[Quarrellama] %s' % llama_msg['content'])
        print('------------------------------------------------------------\n')
        l_full_history.append(llama_msg)

        if agent_choice == 1:
            your_str = input('[You] ')
        else:
            # l_flip_full_history = [{'role': 'user' if msg['role'] == 'assistant' else 'assistant',
            #                         'content': msg['content']}
            #                        for msg in l_full_history]
            flip_llama_msg = {'role': 'user' if llama_msg['role'] == 'assistant' else 'assistant',
                              'content': llama_msg['content']}
            opponent_response = llm.create_chat_completion(
                [flip_llama_msg, sys_msg],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                repeat_penalty=repeat_penalty
            )
            opponent_msg = opponent_response['choices'][0]['message']
            opponent_msg['role'] = 'user'
            l_full_history.append(opponent_msg)
            your_str = opponent_msg['content']
            print('[Quarrelalpaca] %s' % your_str)
        print('------------------------------------------------------------\n')
        if agent_choice == 1 and your_str.strip().lower() == 'end':
            break
        your_msg = {'role': 'user', 'content': your_str}

    out_str = '\n'.join(['%s:%s' % (msg['role'], msg['content']) for msg in l_full_history])
    with open('quarrellama_%s.txt' % datetime.strftime(datetime.now(), '%Y%m%d%H%M%S'), 'w+') as out_fd:
        out_fd.write(out_str)
    print('''\n[Quarrellama] All right. You win.''')
    print('------------------------------------------------------------\n')


if __name__ == '__main__':
    quarrel_llm()