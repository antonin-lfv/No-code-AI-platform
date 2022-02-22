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

class CourseForm(FlaskForm):
    title = StringField('Title', validators=[InputRequired(),
                                             Length(min=10, max=100)])
    description = TextAreaField('Course Description',
                                validators=[InputRequired(),
                                            Length(max=200)])
    price = IntegerField('Price', validators=[InputRequired()])
    level = RadioField('Level',
                       choices=['Beginner', 'Intermediate', 'Advanced'],
                       validators=[InputRequired()])
    available = BooleanField('Available', default='checked')



# Test

class NameForm(FlaskForm):
    name = StringField(label='Entrez votre prénom', validators=[DataRequired()])
    submit = SubmitField('Envoyer')

class SelectBox(FlaskForm):
    choix = SelectField(label="Choisissez une option", choices=["oui", "non"], validators=[DataRequired()])
    submit = SubmitField('Choisir')
