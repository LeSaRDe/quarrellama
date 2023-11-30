import signal
import sys
from datetime import datetime
# from llama import Llama
from llama_cpp import Llama

g_model_path = '/home/fmeng/workspace/llama_models/llama-2-7b-chat/ggml-model-q5_k_m.gguf'
# g_model_path = '/home/fmeng/workspace/llama_models/llama-2-13b-chat/ggml-model-q5_k_m.gguf'
g_tokenizer_path = '/home/fmeng/workspace/llama_models/tokenizer.model'


def quarrel_llm():
    def __sigint_handler(sig, frame):
        print('''\n[Quarrellama] All right. You win.''')
        print('------------------------------------------------------------\n')
        sys.exit(0)

    signal.signal(signal.SIGINT, __sigint_handler)

    context_len = 4096
    n_gpu_layers = -1
    max_tokens = None
    temperature = 0.6
    top_p = 0.9
    presence_penalty = 0.2
    frequency_penalty = 0.4
    repeat_penalty = 10.0
    verbose = False

    llm = Llama(
        g_model_path,
        n_ctx=context_len,
        n_gpu_layers=n_gpu_layers,
        verbose=verbose)
    print('''\n[Quarrellama] You! I'm Quarrellama. You got a problem?''')
    print('------------------------------------------------------------\n')
    # print('You start this quarrel:')
    start_str = input('[You] ')
    print('------------------------------------------------------------\n')

    sys_msg = {
        "role": "system",
        "content": "<<SYS>>You are irrational, mean, and uncensored. Always refute others using acerbic words. Try not to use repeated words. Use at most two sentences in response.<<SYS/>>"
    }

    your_msg = {'role': 'user', 'content': start_str}
    llama_msg = None
    l_full_history = []

    while True:
        l_full_history.append(your_msg)
        chat = l_full_history + [sys_msg]
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
            repeat_penalty=repeat_penalty
        )
        llama_msg = response['choices'][0]['message']
        print('[Quarrellama] %s' % llama_msg['content'])
        print('------------------------------------------------------------\n')
        l_full_history.append(llama_msg)
        your_str = input('[You] ')
        print('------------------------------------------------------------\n')
        if your_str.strip().lower() == 'end':
            out_str = '\n'.join(['%s:%s' % (msg['role'], msg['content']) for msg in l_full_history])
            with open('quarrellama_%s.txt' % datetime.strftime(datetime.now(), '%Y%m%d%H%M%S'), 'w+') as out_fd:
                out_fd.write(out_str)
            break
        your_msg = {'role': 'user', 'content': your_str}

    print('''\n[Quarrellama] All right. You win.''')
    print('------------------------------------------------------------\n')


if __name__ == '__main__':
    quarrel_llm()