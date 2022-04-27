"""
    Due to the method of distant supervision data generation,
    we need at least 4 tasks in a group if we want to reserve some
    tasks for test dataset. So to avoid crash we have to manually select 
    the tasks we want to reserve for test dataset (hide in train).
"""
TEST_TASKS={
    "social_i_qa",
    "qasc",
    "sciq",
    "multi_news",
    "samsum",
    "adversarial_qa-dbidaf",
    "ropes"
}