from framework3.container import Container
from framework3.base.base_clases import BaseFilter
from framework3 import XYData


@Container.bind()
class TWECFilter(BaseFilter):
    def fit(self, x: XYData, y: XYData | None) -> float | None: ...
    def predict(self, x: XYData) -> XYData:
        return XYData.mock((lambda x: 1 if any(list(x.value)) else 0, x.value.index(1)))
