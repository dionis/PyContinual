###---------------------------------------------------------------
### https://huggingface.co/PlanTL-GOB-ES/roberta-base-bne
###----------------------------------------------------------------

from transformers import pipeline
from pprint import pprint
unmasker = pipeline('fill-mask', model='PlanTL-GOB-ES/roberta-base-bne')
pprint(unmasker("Gracias a los datos de la BNE se ha podido <mask> este modelo del lenguaje."))


from transformers import RobertaTokenizer, RobertaModel
tokenizer = RobertaTokenizer.from_pretrained('PlanTL-GOB-ES/roberta-base-bne')
model = RobertaModel.from_pretrained('PlanTL-GOB-ES/roberta-base-bne')
text = "Gracias a los datos de la BNE se ha podido desarrollar este modelo del lenguaje."
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)
print(output.last_hidden_state.shape)
