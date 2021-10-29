
import matplotlib.pyplot as plt
import matplotlib.collections as mcoll
from collections import defaultdict
from matplotlib import colors
from matplotlib.lines import Line2D

import warnings
import seaborn as sns
import math
import numpy as np
import pandas as pd

def deCasteljau(b, t):
    N = len(b)
    a = np.copy(b)
    for r in range(1, N):
        a[:N-r, :] = (1-t)*a[:N-r, :] + t*a[1:N-r+1, :]
    return a[0, :]


def BezierCv(b, nr=5):
    t = np.linspace(0, 1, nr)
    return np.array([[deCasteljau(b, t[k]),
                      deCasteljau(b, t[k+1])] for k in range(nr-1)])


def position_circle(x, radius=1):
    """ Return the x,y coordinate of the point at
        angle (360*x)°, in the circle of radius "radius"
        and center (0, 0)
    """
    return np.array([radius*math.cos(x*2*math.pi),
                     radius*math.sin(x*2*math.pi)])


def linear_gradient(start, end, n=10):
    """ Return a gradient between start and end,
        with n elements.
    """
    gradients = np.zeros((len(start), n))
    for i in range(len(start)):
        gradients[i, :] = np.linspace(start[i], end[i], num=n)
    return np.transpose(gradients)


def linear_gradient_color(c1, c2, n=10):
    """ Return a gradient between the two color c1 & c2
    """
    return linear_gradient(colors.to_rgba(c1), colors.to_rgba(c2), n=n)


def draw_chord(A, B, ax=None, color_start="b", color_end="r",
               precision=1000, **kwargs):
    """ Draw a Bezier curve between two points """
    d = np.linalg.norm(np.array(A) - np.array(B))

    # Depending on the distance between the points
    # the Bezier curve parameters change
    b = [A, A/(1 + d), B/(1 + d), B]
    bz = BezierCv(b, nr=precision)

    lc = mcoll.LineCollection(bz,
                              colors=linear_gradient_color(color_start,
                                                           color_end,
                                                           n=precision), **kwargs)
    ax.add_collection(lc)


def draw_arc_circle(start, end, color="b", radius=1, ax=None,
                    thickness=0.1, precision=1000, **kwargs):
    ts = np.linspace(start, end, precision)
    poly_nodes = ([position_circle(t, radius=radius) for t in ts] +
                  [position_circle(t, radius=radius+thickness)
                   for t in ts[::-1]])
    x, y = zip(*poly_nodes)
    ax.fill(x, y, color=color, **kwargs)


def add_text_circle(x, txt, radius=1, ax=None, **kwargs):
    """ Add text on the border of the circle, in the right orientation """
    ax.text(*position_circle(x, radius=radius),
            txt, rotation=(360*x - 180 if 0.25 < x < 0.75 else 360*x),
            ha='right' if 0.75 > x > 0.25 else 'left',
            va='top' if 0.75 > x > 0.25  else 'bottom',
            rotation_mode='anchor', **kwargs)





class Chords:
    def __init__(self, data, order_col, pair_col, color_col=None,
                 layout_args={}, text_args={},
                 chords_args={}, palette=sns.color_palette()):

        if 'spacing' not in layout_args:
            layout_args['spacing'] = 0
        if 'precision_chord' not in layout_args:
            layout_args['precision_chord'] = 100
        if 'precision_circle' not in layout_args:
            layout_args['precision_circle'] = 100
        if 'thickness_circle' not in layout_args:
            layout_args['thickness_circle'] = 0.1
        if 'subcircle' not in layout_args:
            layout_args['subcircle'] = True
        if 'radius_subcircle' not in layout_args:
            layout_args['radius_subcircle'] = 1.14
        if 'radius_circle' not in layout_args:
            layout_args['radius_circle'] = 1.02
        if 'thickness_subcircle' not in layout_args:
            layout_args['thickness_subcircle'] = 0.1
        if 'internal_chords' not in layout_args:
            layout_args['internal_chords'] = False
        if 'radius_text' not in layout_args:
            layout_args['radius_text'] = max(layout_args['thickness_subcircle']
                                             + layout_args['radius_subcircle'],
                                             layout_args['thickness_circle']
                                             + layout_args['radius_circle']) + 0.1
        if 'no_chords' not in layout_args:
            layout_args['no_chords'] = False
        if 'inverted_grad' not in layout_args:
            layout_args['inverted_grad'] = True
        if 'circle_args' not in layout_args:
            layout_args['circle_args'] = {}
        if 'subcircle_args' not in layout_args:
            layout_args['subcircle_args'] = {}
        if 'singleton' not in layout_args:
            layout_args['singleton'] = True

        if 'palette' not in text_args:
            if color_col is None:
                text_args['palette'] = palette
            else:
                text_args['palette'] = defaultdict(lambda: 'k')


        if not np.all(data[pair_col].value_counts() <= 2):
            raise TypeError("Every value in the `pair` column "
                            "should appear exactly twice")


        if not layout_args['singleton']:
            self.data = data[data.pair_col.map(data.pair_col.value_counts()) == 2]
        else:
            self.data = data.copy()


        self.chords = []
        self.data = data.copy()
        self.order_col = order_col
        self.pair_col = pair_col
        if color_col is None:
            color_col = order_col
        self.color_col = color_col
        self.df = None
        self.layout = layout_args
        self.text_args = text_args
        self.chords_args = chords_args
        self.palette = palette

        self.order_data("order", "pair")
        self.compute_positions()
        self.pair_chords()

    def order_data(self, categories, pairs):
        """
        Return a correctly ordered dataframe, ready to be plotted in chord format
        @ Args:
        data: pd.DataFrame() to reorder, with a column
        `categories` and a column `pair`
        """
        self.format_data()
        self.df["associate_cat_order"] = self.df.apply(
            lambda r: (len(self.mapcat)+r["nbcat"]-r["associate_nbcat"]) % len(self.mapcat)
            + (len(self.mapcat)//2+1 if r["nbcat"]==r["associate_nbcat"] else 0.5),
            axis=1)
        self.df["sort_order"] = self.df.apply(
            lambda r: (r["idx"] if r["nbcat"] <= r["associate_nbcat"]
                       else -r["associate"]), axis=1)
        sign = lambda x: 0 if x == 0 else 1 if x > 0 else -1
        self.df["singleton_sort"] = self.df.apply(lambda r: 0 if r["nbcat"] != r["associate_nbcat"]
                                                  else sign(r["idx"] - r["associate"]), axis=1)
        self.df["internal_sort"] = self.df.apply(lambda r: (0 if r["nbcat"] != r["associate_nbcat"]
                                      else (r["idx"] if r["idx"] < r["associate"]
                                       else -r["associate"])), axis=1)
        self.df = self.df.sort_values(by=["nbcat", "associate_cat_order", "singleton_sort",
                                          "internal_sort", "sort_order"])


    def format_data(self):
        """
        Process the dataframe so that it can be plotted in chord format
        @ Args:
        data: pd.DataFrame() to reorder, with a column
        `categories` and a column `pair`
        """
        if self.color_col == self.order_col:
            self.df = self.data[[self.order_col, self.pair_col]].rename({
                self.pair_col: "pair",
                self.order_col: "order"}, axis=1).copy()
            self.df["color"] = self.df["order"]
        else:
            self.df = self.data[[self.order_col, self.pair_col, self.color_col]].rename({
                self.pair_col: "pair",
                self.order_col: "order",
                self.color_col: "color"}, axis=1).copy()

        self.df.index.names = ['og_idx']
        self.df = self.df.reset_index()

        catunique = self.df["order"].unique()
        self.mapcat = dict(zip(catunique, range(len(catunique))))
        colorunique = self.df["color"].unique()
        self.mapcolor = dict(zip(colorunique, range(len(colorunique))))
        self.df["nbcat"] = self.df["order"].map(self.mapcat).astype(int)
        self.df["nbcolor"] = self.df["color"].map(self.mapcolor).astype(int)
        self.df["idx"] = self.df.index

        pairmap = self.df.groupby("pair").idx.apply(list).to_dict()
        self.df["associate"] = self.df.apply(
            lambda r: [a for a in pairmap[r["pair"]] if a != r["idx"]][0]
            if len(pairmap[r["pair"]]) > 1 else pairmap[r["pair"]][0],
            axis=1)

        self.df["associate_nbcat"] = self.df.associate.map(self.df.nbcat)
        self.df["associate_nbcolor"] = self.df.associate.map(self.df.nbcolor)
        self.df["catcolor"] = self.df.apply(lambda r: (r["nbcat"], r["nbcolor"]),
                                            axis=1
                                           ).astype('category').cat.codes
        self.df["associate_catcolor"] = self.df.apply(lambda r:
                                                      (r["associate_nbcat"], r["associate_nbcolor"]),
                                                      axis=1
                                                     ).astype('category').cat.codes


    def compute_positions(self):
        cat_jump = list(np.where(self.df.nbcat.values[:-1]
                                 != self.df.nbcat.values[1:])[0])
        x = 0
        positions = []
        for i in range(len(self.df)):
            positions.append(x)
            if i in cat_jump:
                x += self.layout['spacing']
            x += (1 - self.layout['spacing']*(len(cat_jump)+1))/(len(self.df))
        self.df["position"] = positions
        self.df["associate_position"] = self.df.associate.map(self.df.position)

    def pair_chords(self):
        self.chords = list(zip(
            self.df.position,
            self.df.associate_position,
            self.df.nbcolor,
            self.df.associate_nbcolor,
            self.df.nbcat,
            self.df.associate_nbcat))

        # add chord except if singleton
        self.chords = [tpl for tpl in self.chords if tpl[0] != tpl[1]]

    def add_chord(self, idx1, idx2):
        dd = self.df.set_index("og_idx")
        self.chords.append(
        (dd.loc[idx1].position, dd.loc[idx2].position,
         dd.loc[idx1].nbcolor, dd.loc[idx2].nbcolor,
         dd.loc[idx1].nbcat, dd.loc[idx2].nbcat))

    def plot(self, ax=None):
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 8))
            ax.axis('off')

        nb_to_name_cat = {self.mapcat[k]:k for k in self.mapcat}
        positions = self.df.position.values
        catcolors = self.df.catcolor.values
        idxs = np.where(catcolors[:-1] != catcolors[1:])[0]
        start_categorie = [0] + list(positions[idxs+1])
        end_categorie = list(positions[idxs]) + [positions[-1]]
        cats = [self.df.nbcolor.iloc[0]] + list(self.df.nbcolor.iloc[idxs+1])

        for s, e, c in zip(start_categorie, end_categorie, cats):
            draw_arc_circle(s - 0.5/len(self.df), e + 0.5/len(self.df),
                            color=self.palette[c], ax=ax,
                            precision=self.layout['precision_circle'],
                            thickness=self.layout['thickness_circle'],
                            radius=self.layout['radius_circle'],
                            **self.layout['circle_args'])

        # the radius text should correspond to categories
        cats = self.df.nbcat.values
        idxs = np.where(cats[:-1] != cats[1:])[0]
        start_categorie = [0] + list(positions[idxs+1])
        end_categorie = list(positions[idxs]) + [positions[-1]]
        cats = [cats[0]] + list(cats[idxs+1])
        for s, e, c in zip(start_categorie, end_categorie, cats):
            add_text_circle((s + e - 1/len(self.df))/2, nb_to_name_cat[c], ax=ax,
                            color=self.text_args['palette'][c],
                            radius=self.layout['radius_text'],
                            **{k: v for k, v in self.text_args.items() if k != 'palette'})

        if self.layout['subcircle']:
            catcolors = self.df.associate_catcolor.values
            idxs = np.where(catcolors[:-1] != catcolors[1:])[0]
            start_subcategorie = [0] + list(positions[idxs+1])
            end_subcategorie = list(positions[idxs]) + [positions[-1]]
            subcats = [self.df.associate_nbcolor.iloc[0]] + list(self.df.associate_nbcolor.iloc[idxs+1])

            for s, e, c in zip(start_subcategorie, end_subcategorie, subcats):
                draw_arc_circle(s - 0.5/len(self.df), e + 0.5/len(self.df),
                                color=self.palette[c], ax=ax,
                                precision=self.layout['precision_circle'],
                                thickness=self.layout['thickness_subcircle'],
                                radius=self.layout['radius_subcircle'],
                                **self.layout['subcircle_args'])

        if not self.layout['no_chords']:
            for pos_1, pos_2, color_1, color_2, cat_1, cat_2 in self.chords:
                if cat_1 != cat_2 or self.layout['internal_chords']:
                    draw_chord(position_circle(pos_2),
                               position_circle(pos_1), ax=ax,
                               color_start=self.palette[color_2
                                                   if self.layout['inverted_grad']
                                                   else color_1],
                               color_end=self.palette[color_1
                                                 if self.layout['inverted_grad']
                                                 else color_2],
                               precision=self.layout['precision_chord'],
                               **self.chords_args)

        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.axis('equal')
        ax.axis('off')
        return ax








def chord_diagram(categories, pairs, hues=None,
                  data=None, ax=None, palette=sns.color_palette(),
                  layout_args={}, text_args={}, chord_args={}):
    """ Draw a chord diagram.
        @ Args
        - categories: Categories of each individual.
        Decide the order of the plot.
        Either a list or a column name if `data` is not None.
        - pair: For each individual identifies the pair it's in.
        Every value should appear twice. Either a list or a column
        name if `data` is not None.
        - hues: list of categories that will determine the color of the plot.
        - data: dataset containing the columns `categories` and `pair`
        - ax: matplotlib ax object
        - palette: seaborn palette
        - layout_args: dict arguments for the layout, include:
            * 'spacing' (default 0): space between the categories
            * precision_chord: precision to plot the chord,
                               higher = better but slower.
            * precision_circle: same for the circles
            * subcircle: presence or not of a subcircle (see examples)
            * thickness_circle, thickness_subcircle: width of the circle / subcircle (default 0.1)
            * radius_circle, radius_subcircle: radii of both circles
            * internal_chords: Plot or not the internal chords (default `False`)
            * radius_text: radius of the text
            * no_chords: Don't plot the chords (good for testing, default `False`)
            * inverted_grad: Inverse the gradient on the chords (default `True`)
            * circle_args / subcircle_args: dict, default `{}`, additional arguement for ax.fill
            * nuplets: allow for more than one link going from the same node. Default `False`
            * singletons: allow for nodes with no "paired" value, default `True`
            * plot: Default `True`, if `False` does not plot the figure.
    """
    if 'nuplets' not in layout_args:
        layout_args['nuplets'] = False
    if 'plot' not in layout_args:
        layout_args['plot'] = True
    if layout_args['nuplets']:
        layout_args['singletons'] = True

    if data is None:
        data = pd.DataFrame()
        data["cat"] = categories
        data["pair"] = pairs
        data["col"] = hues
        categories = "cat"
        pairs = "pair"
        hues = None if hues is None else "col"

    doublets = None
    if layout_args['nuplets']:
        data_copy = data.copy()
        data_copy.index.names = ['idx']
        data_copy = data_copy.reset_index()
        map_pair = data_copy.groupby(pairs).idx.apply(list)
        new_pairs = [str(p) + str(map_pair[p].index(idx)//2)
                    for idx, p in zip(data_copy.idx, data_copy[pairs])]
        data_copy[pairs] = new_pairs
    else:
        data_copy = data.copy()

    ch = Chords(data=data_copy, order_col=categories,
                pair_col=pairs,
                color_col=hues,
                 layout_args=layout_args, text_args=text_args,
                 chords_args=chord_args, palette=palette)

    if layout_args['nuplets']:
        for idx1, p1, p2 in zip(data.index, data[pairs], data_copy[pairs]):
            if len(map_pair) > 2:
                for idx2 in map_pair[p1]:
                    ch.add_chord(idx1, idx2)

    if layout_args['plot']:
        ch.plot(ax=ax)
    return ch
