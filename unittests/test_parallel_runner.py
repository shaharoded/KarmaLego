import pytest
import json
import pandas as pd
from core.parallel_runner import run_parallel_jobs

def test_run_parallel_jobs_simple(tmp_path):
    """
    Test that run_parallel_jobs correctly executes multiple jobs
    and aggregates the results with a 'job_name' column.
    """
    # 1. Setup dummy data
    # Cohort 1: A -> B
    p1 = [(0, 1, "A"), (2, 3, "B")]
    data1 = [p1, p1] # Duplicate to ensure support
    
    # Cohort 2: C -> D
    p2 = [(0, 1, "C"), (2, 3, "D")]
    data2 = [p2, p2]
    
    # 2. Setup dummy inverse map file (required by discover_patterns)
    map_file = tmp_path / "test_map.json"
    with open(map_file, "w") as f:
        json.dump({}, f)
    str_map_path = str(map_file)

    # 3. Define jobs
    jobs = [
        {
            'name': 'job_AB',
            'data': data1,
            'params': {
                'epsilon': 0, 
                'max_distance': 10, 
                'min_ver_supp': 0.5, 
                'min_length': 2,
                'inverse_mapping_path': str_map_path
            }
        },
        {
            'name': 'job_CD',
            'data': data2,
            'params': {
                'epsilon': 0, 
                'max_distance': 10, 
                'min_ver_supp': 0.5, 
                'min_length': 2,
                'inverse_mapping_path': str_map_path
            }
        }
    ]

    # 4. Run with 2 workers
    result_df = run_parallel_jobs(jobs, num_workers=2)

    # 5. Assertions
    assert not result_df.empty
    assert "job_name" in result_df.columns
    
    # Verify both jobs are present
    unique_jobs = sorted(result_df["job_name"].unique())
    assert unique_jobs == ["job_AB", "job_CD"]
    
    # Verify content of job_AB
    df_ab = result_df[result_df["job_name"] == "job_AB"]
    # Check for pattern (A, B)
    # symbols are tuples in the dataframe
    found_ab = any(tuple(row.symbols) == ("A", "B") for row in df_ab.itertuples())
    assert found_ab, "Job AB should have found pattern A->B"

    # Verify content of job_CD
    df_cd = result_df[result_df["job_name"] == "job_CD"]
    found_cd = any(tuple(row.symbols) == ("C", "D") for row in df_cd.itertuples())
    assert found_cd, "Job CD should have found pattern C->D"
