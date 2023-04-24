import pandas as pd
from sklearn.neighbors import NearestNeighbors


class NearestNeighboursQuery:

    metadata: pd.DataFrame
    neighbours: int
    nn: NearestNeighbors
    coordinates: pd.DataFrame

    def __init__(
            self,
            coordinates: pd.DataFrame,
            metadata: pd.DataFrame,
            neighbours: int = 6,
    ):
        self.metadata = metadata
        self.neighbours = neighbours
        self.coordinates = coordinates
        self.__setup()

    def __setup(self):
        self.nn = NearestNeighbors(n_neighbors=self.neighbours)
        self.nn.fit(self.coordinates)

    def search(self, search_term: str, column: str) -> pd.DataFrame:
        search_idx = self.get_search_index(search_term, column)
        return self.get_nearest_neighbours(search_idx)

    def get_search_index(self, search_term: str, column: str) -> int:
        return self.metadata[self.metadata[column] == search_term].index[0]

    def get_nearest_neighbours(
            self,
            search_idx: int,
    ) -> pd.DataFrame:
        search_item = pd.DataFrame([self.coordinates.iloc[search_idx]])
        distances, nn_indices = (a.T for a in self.nn.kneighbors(search_item, self.neighbours))
        nearest_metadata = self.metadata.iloc[nn_indices.T[0]]
        nearest_metadata.insert(0, 'distance', distances)
        return nearest_metadata
