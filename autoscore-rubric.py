# from rubrics import get_rubric_df
# from request import make_formatted_request
# from type import StructuredComponentRepresentation


# if __name__ == "__main__": 
#   RUBRIC_PATH = "open-ended-response.csv"
#   rubric_df = get_rubric_df(RUBRIC_PATH)
#   columns = rubric_df.columns.to_list()

#   for index, row in rubric_df.iterrows():
    
#     pairs = zip(columns, row)
#     category_dict = dict(pairs)

#     # pass to llm to craft a Structured Component Representation
#     prompt = f"This is a Python dictionary representation of a rubric criteria row: {category_dict}. \
#         Translate the row and score requirements into specific features that a teacher can look for in a \
#         student's response. For example, rubric may score on the correctness of a student's response, and \
#         therefore a feature could be 'feature_name': 'Correct information', 'data_type': boolean. Use simple \
#         types, like boolean, string, integers, floats, or lists of these types. Do not use data types that \
#         correspond directly to scores. String data types should be Your job is to create an evidence \
#         collection guide based on the rubric row for another evaluator to fill in with evidence."
#     res = make_formatted_request(prompt, StructuredComponentRepresentation)

#     print(res.output_text)

#     break


# # Maybe it would be better to pass the original rubric row into agent 1 and allow it to collect evidence 
# # based on the original?