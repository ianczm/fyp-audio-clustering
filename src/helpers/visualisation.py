import pandas as pd
import math
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go


class Visualiser:

    name: str

    def __init__(self, name: str):
        self.name = name

    def x_name(self):
        return self.name + '1'

    def y_name(self):
        return self.name + '2'

    def generate_plot_coloured_by_features(self, dataframe: pd.DataFrame, color_by: str):
        px.scatter(
            dataframe,
            x=self.x_name(),
            y=self.y_name(),
            hover_data=['artist', 'song_name'],
            template='plotly_dark',
            color=color_by
        ).show()

    @staticmethod
    def generate_hover_template(field_names: list[str]):
        return '<br>'.join([f'<b>{name}</b>: %{{customdata[{idx}]}}' for idx, name in enumerate(field_names)]) + '<extra></extra>'

    def generate_colour_feature_trace(self, fig: go.Figure, dataframe: pd.DataFrame, feature: str, row: int, col: int):
        fig.add_trace(go.Scatter(
            x=dataframe[self.x_name()],
            y=dataframe[self.y_name()],
            customdata=dataframe[['playlist', 'artist', 'song_name']].to_numpy(),
            mode='markers',
            hovertemplate=self.generate_hover_template(['playlist', 'artist', 'song_name']),
            showlegend=False,
            marker={'color': dataframe[feature], 'colorscale': 'Agsunset'},
            name=feature),
            row=row,
            col=col
        )

    def generate_subplot_colour_all_features(self, dataframe: pd.DataFrame, features: list[str], cols: int = 3):
        blacklisted_features = ['playlist', 'artist', 'song_name', self.x_name(), self.y_name()]
        rows = math.ceil(len(features) / cols)
        fig = make_subplots(rows=rows, cols=cols, subplot_titles=(
            tuple(feature for feature in features if feature not in blacklisted_features)
        ))
        idx = 0
        for feature in features:
            if feature not in blacklisted_features:
                self.generate_colour_feature_trace(
                    fig,
                    dataframe,
                    feature,
                    row=(math.floor(idx/cols)+1),
                    col=(idx%cols)+1
                )
                idx += 1
        fig.update_layout(title=f'{self.name} Result Coloured by Features', height=350*rows, template='plotly_dark')
        fig.update_annotations(font={'size': 12})
        fig.show()

    def generate_tsne_trace(self, fig, tsne, row, col, playlists, artists, song_names):
        tsne['playlist'] = playlists
        tsne['artist'] = artists
        tsne['song_name'] = song_names
        for pl in pd.unique(tsne['playlist']):
            filtered_tsne = tsne[tsne['playlist'] == pl]
            fig.add_trace(go.Scatter(
                x=filtered_tsne[self.x_name()],
                y=filtered_tsne[self.y_name()],
                customdata=filtered_tsne.iloc[:, -3:].to_numpy(),
                mode='markers',
                hovertemplate=self.generate_hover_template(['playlist', 'artist', 'song_name']),
                showlegend=False,
                name=pl),
                row=row,
                col=col
            )

    def generate_tsne_perplexity_subplots(self, tsne_array, cols, playlists, artists, song_names):
        rows = math.ceil(len(tsne_array) / cols)
        fig = make_subplots(rows=rows, cols=cols, subplot_titles=(tuple(f'Perplexity: {p}, Exaggeration: {e}' for e, p, _ in tsne_array)))

        for idx, item in enumerate(tsne_array):
            ee, perplexity, tsne = item
            self.generate_tsne_trace(
                fig,
                tsne,
                row=(math.floor(idx/cols)+1),
                col=(idx%cols)+1,
                playlists=playlists,
                artists=artists,
                song_names=song_names
            )

        fig.update_layout(title=f'{self.name} Perplexities', height=350*rows, template='plotly_dark')
        fig.update_annotations(font={'size': 12})
        fig.show()


def plot_subplots(options, title: str, cols: int = 2, col_for_each_feature: bool = True):
    def add_trace(fig, coordinates, feature, metadata, row, col):
        fig.add_trace(
            go.Scatter(
                x=coordinates.iloc[:, 0],
                y=coordinates.iloc[:, 1],
                customdata=metadata.to_numpy(),
                mode='markers',
                hovertemplate=Visualiser.generate_hover_template(['playlist', 'artist', 'song_name', 'cluster']),
                showlegend=False,
                marker={
                    'color': metadata[feature].astype('category').cat.codes,
                    'colorscale': 'portland' if feature == 'cluster' else 'spectral'
                },
                name=feature
            ), row=row, col=col
        )

    rows = len(options) if col_for_each_feature else math.ceil(len(options) / cols)

    subplot_titles = [f'{heading}, {feature}' for _,_,features,heading in options for feature in features]
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=subplot_titles)

    for idx, option in enumerate(options):
        coordinates, metadata, features, heading = option
        for col, feature in enumerate(features):
            add_trace(
                fig,
                coordinates,
                feature,
                metadata,
                idx+1 if col_for_each_feature else (math.floor(idx/cols)+1),
                col+1 if col_for_each_feature else (idx % cols)+1
            )

    fig.update_layout(title=title, height=350*rows, template='plotly_dark')
    fig.update_annotations(font={'size': 12})
    fig.show()




