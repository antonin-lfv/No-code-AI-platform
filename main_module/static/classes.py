from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, SelectField, SelectMultipleField, FieldList
from wtforms.validators import DataRequired

label_dataset = "Dataset"
choix_dataset = ['-- Choisissez une option --', 'Iris (Classification)', 'Penguins (Classification)', 'Prix des voitures (Régression)', 'Choisir un dataset personnel']

class SelectDataset(FlaskForm):
    choix = SelectField(label=label_dataset, choices=choix_dataset, validators=[DataRequired()], default=choix_dataset[0])
    submit = SubmitField('Choisir')









# Test

class NameForm(FlaskForm):
    name = StringField(label='Entrez votre prénom', validators=[DataRequired()])
    submit = SubmitField('Envoyer')

class SelectBox(FlaskForm):
    choix = SelectField(label="Choisissez une option", choices=["oui", "non"], validators=[DataRequired()])
    submit = SubmitField('Choisir')
