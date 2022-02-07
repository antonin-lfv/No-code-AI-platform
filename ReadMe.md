### Python inside html template
```html
{% i=i+1 %}
```

### get config data
```html
<p>Valeur stockée dans config.py : {{ config['CONSTANTE'] }}</p>
```

### use parameter of root function
```html
<p>Je suis {{ user_name }}</p>
```

with
```py
@app.route('/result')
def result():
    return render_template('result.html', user_name="Anto")
```
