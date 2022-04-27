from promptsource.templates import get_templates_data_frame

# This df is big so we better cache this
meta_data = get_templates_data_frame()

def query_metadata(qtext):
    """
        Returns the row in promptsource metadata dataframe that contains all the metadata of the dataset/subset/template specified
        by qtext.

        Available Metadata Columns: 
            reference, original_task, choices_in_prompt, metrics, answer_choices, jinja
    """
    ds_fullname = qtext.split("|")[0]
    if "-" not in ds_fullname:
        subset = None
        dataset = ds_fullname
    else:
        dataset = ds_fullname.split("-")[0]
        subset = ds_fullname.split("-")[1]
    template = qtext.split("|")[2]
    if subset:
        return meta_data.loc[(meta_data['dataset'] == dataset) & (meta_data['subset'] == subset) & (meta_data['name'] == template)]
    else:
        return meta_data.loc[(meta_data['dataset'] == dataset) & (meta_data['name'] == template)]

if __name__ == "__main__":
    print(query_metadata("emotion|xxxx|answer_with_class_label|3ffff"))