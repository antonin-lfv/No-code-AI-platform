import wtforms.widgets
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, SelectField, SelectMultipleField, FieldList, IntegerField, BooleanField, TextAreaField, RadioField
from wtforms.validators import InputRequired, Length, DataRequired

"""
StringField: A text input.
TextAreaField: A text area field.
IntegerField: A field for integers.
BooleanField: A checkbox field.
RadioField: A field for displaying a list of radio buttons for the user to choose from.
"""

label_dataset = "Dataset"
choix_dataset = ['-- Choisissez une option --', 'Iris (Classification)', 'Penguins (Classification)', 'Prix des voitures (Régression)']  #, 'Choisir un dataset personnel']
label_colonnes = "Choisissez des colonnes"

class SelectDataset(FlaskForm):
    """ Page Dataset """
    choix = SelectField(label=label_dataset, choices=choix_dataset, validators=[InputRequired()], default=choix_dataset[0])
    submit = SubmitField('Choisir')

