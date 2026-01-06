# MATU

This is the sample code for running camel framework on GPT-4o and the MATH Dataset. Please make sure to install ```camel-ai```, ```openai```, ```tensorly```, ```scikit-learn``` and ```sentence_transformers``` package and prepare ready for the MATH Dataset. Also change the api key in the ```camel_math.py```. 

## Running Pipeline

Please first run ```camel_math.py``` to get the accuracy dict, embedding and conversation log. Then run ```cp2.py``` to get the fit_dict and run ```inter_unq.py``` to get the final uncertainty value. Finally, running ```evaluation.py``` to output the evaluation results.