# Chordialement

![Example of a chord diagram](test/example_chord.png)

Chord diagrams ('circos') can be used to visualize a very specific type of dataset. The data should contain observations that fall in discrete categories and have pairwise, but ideally bidirectional, association between observations. Say a dataset containing information about twins' political opinions, or couples favourite fruits.  

In most cases, these plots don't bring more information than a triangular [jointplot](https://seaborn.pydata.org/generated/seaborn.jointplot.html). But they are arguably prettier and allow to show individual observation. It can also be easier to add more information to the plot (by modifying the width & color of the link for example). They can also deal (badly) with the occasional triplet/quadruplet.  

This package is another attempt to make them easy to plot. Similar tentatives include [Circos](http://circos.ca), [Chord](https://pypi.org/project/chord/), [Bokeh](https://bokeh.org) and [plotly](https://plotly.org). I wanted a pure python one, based on matplotlib, and as customizable as possible, so this is it.

## Install

``` sh
git clone git@github.com:Thopic/chordialement.git
pip install -r .
```

## Example

Everything is in the `chord_diagram` function. By default, the colors are
defined by the categories, but they can also be separated. Both functions return 
a `Chords` object that can be manipulated to some extent.

``` python
import pandas as pd
import numpy as np
from chordialement import chord_diagram, colored_chords

rng = np.random.default_rng()
df = pd.DataFrame()
df["favourite_fruit"] = rng.choice(["Apple", "Orange", "Kiwi", "Tomato", "Banana"], size=200)
df["couple"] = rng.choice(list(range(100))*2, size=200, replace=False)
df["Like Potatoes"] = rng.choice([True, False], size=200)
ch1 = chord_diagram(categories="favourite_fruit", pair="couple", 
                    layout_args={'spacing': 0.01, 'internal_chords': True},
                    data=df)
ch2 = chord_diagram(categories="favourite_fruit", pair="couple", hues="Like Potatoes" 
                    layout_args={'spacing': 0.01, 'internal_chords': True},
                    data=df)
```
![Example of a chord diagram](test/example_chord.png)
![Example of a colored chord diagram](test/example_chord_2.png)


## Singletons & Triplets

Singletons are fairly straightforward and dealt with by default (don't forget to set `internal_chords` to `True` so that duplets in the same categories are different from singletons).

Triplets are a bit more complicated, as there's no good (well, simple) way of ordering them, the way they're plotted depend a lot on the ordering of the initial dataframe, so you can try to play with that if the results are not convincing.

``` python
import pandas as pd
import numpy as np

df = pd.DataFrame()
df["favourite_fruit"] = rng.choice(["Apple", "Orange", "Kiwi", "Tomato", "Banana"], size=165)
df["couple"] = rng.choice(list(range(50))*2 + list(range(50, 100)) + list(range(100, 105))*3, size=165, replace=False)
ch = chord_diagram(categories="favourite_fruit", pairs="couple", 
                         layout_args={'spacing': 0.01, 'internal_chords': True, 'nuplets': True},
                         data=df)
```
Additionnally, if in need of precise control, you can add new chords manually:

``` python
fig, ax = plt.subplots()
ch = chord_diagram(categories="favourite_fruit", pairs="couple", ax=ax,
                         layout_args={'spacing': 0.01, 'internal_chords': True, 'nuplets': True, 'plot': False},
                         data=df)
for ii in range(4, 12):
    ch.add_chord(1, ii)
_ = ch.plot(ax)

```





