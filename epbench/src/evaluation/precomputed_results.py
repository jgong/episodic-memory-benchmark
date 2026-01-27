from epbench.src.evaluation.evaluation_wrapper import EvaluationWrapper
from epbench.src.evaluation.generator_answers_2_rag import get_top_n
import pandas as pd

def get_precomputed_results(experiments, 
                            env_file, 
                            data_folder,
                            all_benchmarks = {'benchmark_claude_default_20': None,
                                              'benchmark_claude_default_200': None,
                                              'benchmark_claude_default_2000': None,
                                              'benchmark_claude_default_20_ordered': None,
                                              'benchmark_claude_default_200_ordered': None,
                                              'benchmark_gpt_default_20': None,
                                              'benchmark_gpt_default_200': None,
                                              'benchmark_claude_news_20': None,
                                              'benchmark_claude_news_200': None,
                                              'benchmark_claude_scifi_20': None,
                                              'benchmark_claude_scifi_200': None},
                            evaluation_policy = 'remove_duplicates'):
    df_list = []

    for i in range(len(experiments)):
        df_list.append(pd.DataFrame(experiments[i], index=[0]))
    # concatenate all DataFrames in the list
    df = pd.concat(df_list, ignore_index=True)
    df['evaluation_object'] = None

    for i in range(len(df)):
        df_cur = df.iloc[i]

        # Construct benchmark key based on model name, book type, nb_events, and ordered flag
        model_name = df_cur['book_model_name']
        nb_events = df_cur['book_nb_events']
        
        # Determine book type (default, news, scifi)
        book_type = 'default'
        if 'book' in df_cur.index:
            if df_cur['book'] in ['default', 'news', 'scifi']:
                book_type = df_cur['book']
            else:
                raise ValueError(f'Unknown book type: {df_cur["book"]}')
        
        # Check if ordered
        is_ordered = ('ordered' in df_cur.index) and (df_cur['ordered'])
        
        # Build benchmark key
        # Format: benchmark_{model_name}_{book_type}_{nb_events}{_ordered}
        def build_key():
            key = f'benchmark_{model_name}_{book_type}_{nb_events}'
            if is_ordered:
                key += '_ordered'
            return key
        
        # Try primary key
        benchmark_key = build_key()
        
        if benchmark_key not in all_benchmarks:
            raise ValueError(f'Benchmark key `{benchmark_key}` not found in all_benchmarks. '
                           f'Please provide benchmark in all_benchmarks.')
        
        my_benchmark = all_benchmarks[benchmark_key]

        if df_cur['answering_kind'] == 'prompting':
            answering_parameters = {'kind': df_cur['answering_kind'],
                                    'model_name': df_cur['answering_model_name'],
                                    'max_new_tokens': 4096,
                                    'sleeping_time': 1,
                                    'policy': evaluation_policy}           
        elif df_cur['answering_kind'] == 'rag':
            answering_parameters = {'kind': df_cur['answering_kind'], 
                                    'model_name': df_cur['answering_model_name'], 
                                    'max_new_tokens': 4096, 
                                    'sleeping_time': 1, 
                                    'embedding_chunk': df_cur['answering_embedding_chunk'], 
                                    'embedding_model': "text-embedding-3-small", 
                                    'embedding_batch_size': 2048, 
                                    'top_n': get_top_n(df_cur['answering_embedding_chunk'], my_benchmark), 
                                    'policy': evaluation_policy}
        elif df_cur['answering_kind'] == 'ftuning':
            answering_parameters = {'kind': df_cur['answering_kind'], 
                                    'model_name': df_cur['answering_model_name'], 
                                    'max_new_tokens': 4096, 
                                    'sleeping_time': 0, 
                                    'ftuning_input_data_policy': 'single', 
                                    'ftuning_need_upload': False, 
                                    'ftuning_need_actual_tune': False, 
                                    'batch_size': 'auto', 
                                    'learning_rate_multiplier': 'auto', 
                                    'n_epochs': 10,
                                    'policy': evaluation_policy}
            # ad-hoc
            if df_cur['book_nb_events'] == 20:
                if df_cur['answering_model_name'] == 'gpt-4o-mini-2024-07-18':
                    answering_parameters['fine_tuned_model_name'] = 'ft:gpt-4o-mini-2024-07-18:personal::AAzm9XtH'
                elif df_cur['answering_model_name'] == 'gpt-4o-2024-08-06':
                    answering_parameters['fine_tuned_model_name'] = 'ft:gpt-4o-2024-08-06:personal::AB02Cbei'
                else:
                    raise ValueError('only done for gpt4o and gpt4o-mini')
            elif df_cur['book_nb_events'] == 200:
                if df_cur['answering_model_name'] == 'gpt-4o-mini-2024-07-18':
                    answering_parameters['fine_tuned_model_name'] = 'ft:gpt-4o-mini-2024-07-18:personal::AB0B6H4o'
                elif df_cur['answering_model_name'] == 'gpt-4o-2024-08-06':
                    answering_parameters['fine_tuned_model_name'] = 'ft:gpt-4o-2024-08-06:personal::DISCARDED' # DISCARDED (~400 dollars)
                else:
                    raise ValueError('only done for gpt4o and gpt4o-mini')
        str_print = f"Document with {my_benchmark.nb_tokens()} tokens, answer with {df_cur['answering_kind']} using with {df_cur['answering_model_name']}"
        if df_cur['answering_kind'] == 'rag':
            str_print = f"{str_print} ({df_cur['answering_embedding_chunk']} chunks)"
        print(str_print)
        my_evaluation = EvaluationWrapper(my_benchmark, answering_parameters, data_folder, env_file)
        df.loc[i, 'evaluation_object'] = my_evaluation
    return df
