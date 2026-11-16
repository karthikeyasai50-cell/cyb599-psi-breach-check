"""Protocol A: Diffie-Hellman PSI (Part 2, Section 3), built on OpenMined PSI.

OpenMined PSI implements the blinded-exponent protocol on NIST P-256:
the client sends X_i = H(x_i)^a, the server returns Z_i = X_i^b, and the
client unblinds and looks W_i = H(x_i)^b up in the server's published,
Golomb-compressed set of H(s_j)^b.

Every message crosses the boundary as serialized bytes, so the sizes reported
by the benchmarks are the real wire sizes.

Not implemented here (see the report's limitations): the optional
Chaum-Pedersen proof of key consistency, because OpenMined PSI exposes no
hook for it.
"""

import private_set_intersection.python as _psi

from .params import DHParams

_DS = {
    "GCS": _psi.DataStructure.GCS,
    "BLOOM_FILTER": _psi.DataStructure.BLOOM_FILTER,
    "RAW": _psi.DataStructure.RAW,
}


class DHServer:
    """Holds the long-term key b and publishes the blinded, compressed set F."""

    def __init__(self, params: DHParams, items: list):
        self.params = params
        self._server = _psi.server.CreateWithNewKey(True)
        # F is built once for any query of up to max_client_items items, so it
        # never depends on a particular client's k.
        setup = self._server.CreateSetupMessage(
            params.fpr_total, params.max_client_items, items, _DS[params.data_structure])
        self.setup_bytes = setup.SerializeToString()

    def process(self, request_bytes: bytes) -> bytes:
        req = _psi.Request()
        req.ParseFromString(request_bytes)
        return self._server.ProcessRequest(req).SerializeToString()


class DHClient:
    """Uses one fresh scalar a per query (OpenMined PSI's client key)."""

    def __init__(self, setup_bytes: bytes):
        # The client downloads and caches F once per breach-set version.
        self._setup = _psi.ServerSetup()
        self._setup.ParseFromString(setup_bytes)

    def query(self, items: list) -> bytes:
        self._items = list(items)
        self._client = _psi.client.CreateWithNewKey(True)   # fresh a for this query
        return self._client.CreateRequest(self._items).SerializeToString()

    def finish(self, response_bytes: bytes) -> set:
        resp = _psi.Response()
        resp.ParseFromString(response_bytes)
        idx = self._client.GetIntersection(self._setup, resp)
        return {self._items[i] for i in idx}
