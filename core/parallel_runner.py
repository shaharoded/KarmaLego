import pandas as pd
from core.karmalego import KarmaLego
import multiprocessing

def _worker_process(job_info):
    """
    Worker function to run a single KarmaLego job.
    Must be top-level for multiprocessing on Windows.
    """
    name = job_info['name']
    data = job_info['data']
    params = job_info['params']
    
    # Extract core init params
    epsilon = params.get('epsilon')
    max_distance = params.get('max_distance')
    min_ver_supp = params.get('min_ver_supp')
    
    # Initialize algorithm
    kl = KarmaLego(epsilon, max_distance, min_ver_supp)
    
    # Prepare runtime arguments (exclude init params)
    runtime_args = {k: v for k, v in params.items() 
                   if k not in ['epsilon', 'max_distance', 'min_ver_supp']}
    
    print(f"Starting job: {name}")
    # Run discovery (returns tuple, we take the first element: the DataFrame)
    result = kl.discover_patterns(data, **runtime_args)
    
    if isinstance(result, tuple):
        df = result[0]
    else:
        df = result
    
    # Add job identifier
    df['job_name'] = name
    return df

def run_parallel_jobs(jobs_list, num_workers=None):
    """
    Run multiple KarmaLego jobs in parallel and concatenate results.
    
    Parameters
    ----------
    jobs_list : list[dict]
        List of job dictionaries. Each dict must have:
        - 'name': str (Job identifier)
        - 'data': list (Entity list)
        - 'params': dict (All params for KarmaLego init and discover_patterns)
          Example params: {'epsilon': 10, 'max_distance': 100, 'min_ver_supp': 0.5, 'max_length': 3}
    num_workers : int, optional
        Number of parallel processes. Defaults to CPU count.
        
    Returns
    -------
    pd.DataFrame
        Concatenated results from all jobs.
    """
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()
        
    print(f"Running {len(jobs_list)} jobs with {num_workers} workers...")
    
    with multiprocessing.Pool(num_workers) as pool:
        dfs = pool.map(_worker_process, jobs_list)
        
    if not dfs:
        return pd.DataFrame()
        
    final_df = pd.concat(dfs, ignore_index=True)
    return final_df

# # --- Usage Example ---
# if __name__ == "__main__":
#     # Example setup
#     jobs = [
#         {'name': 'cohort_A', 'data': entity_list_A, 'params': {'epsilon': 10, 'max_distance': 100, 'min_ver_supp': 0.5, 'max_length': 3}},
#         {'name': 'cohort_B', 'data': entity_list_B, 'params': {'epsilon': 10, 'max_distance': 100, 'min_ver_supp': 0.4, 'max_length': 3}}
#     ]
#     result_df = run_parallel_jobs(jobs)
#     result_df.to_csv("all_patterns.csv", index=False)
#     pass