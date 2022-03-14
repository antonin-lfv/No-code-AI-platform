<h1 align="center">No Code AI platform</h1>

<br/>

<p align="center">
	<a href="https://no-code-ai-platform.herokuapp.com"><img src="https://img.shields.io/badge/Heroku-430098?style=for-the-badge&logo=heroku&logoColor=white" height="20"/></a>
	<p/>
<br/>


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



## Info boxes
https://getbootstrap.com/docs/4.0/components/alerts/


## Collapse
https://getbootstrap.com/docs/4.0/components/collapse/


## dropdowns
https://getbootstrap.com/docs/4.0/components/dropdowns/


## over texte
https://getbootstrap.com/docs/4.0/components/popovers/
https://getbootstrap.com/docs/4.0/components/tooltips/

## Accordeon
https://getbootstrap.com/docs/5.0/components/accordion/

## Plotly in flask
https://towardsdatascience.com/web-visualization-with-plotly-and-flask-3660abf9c946

## Probably for metrics
https://getbootstrap.com/docs/5.0/components/badge/
https://getbootstrap.com/docs/5.0/utilities/position/