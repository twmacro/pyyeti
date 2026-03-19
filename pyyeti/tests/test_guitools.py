import pytest
from pyyeti import guitools


@pytest.mark.parametrize("read", [True, False])
def test_get_file_name_no_tkinter(read, monkeypatch):
    monkeypatch.setattr(guitools, "HAVE_TKINTER", False)
    with pytest.raises(ImportError, match="tkinter not available.*cannot create GUI"):
        guitools.get_file_name(None, read)


def test_askopenfilename_no_tkinter(monkeypatch):
    monkeypatch.setattr(guitools, "HAVE_TKINTER", False)
    with pytest.raises(ImportError, match="tkinter not available.*cannot create GUI"):
        guitools.askopenfilename()


def test_asksaveasfilename_no_tkinter(monkeypatch):
    monkeypatch.setattr(guitools, "HAVE_TKINTER", False)
    with pytest.raises(ImportError, match="tkinter not available.*cannot create GUI"):
        guitools.asksaveasfilename()


def test_multicolumnlistbox_no_tkinter(monkeypatch):
    monkeypatch.setattr(guitools, "HAVE_TKINTER", False)
    headers = ["First", "Middle", "Last"]
    lst1 = ["Tony", "Jennifer", "Albert", "Marion"]
    lst2 = ["J.", "M.", "E.", "K."]
    lst3 = ["Anderson", "Smith", "Kingsley", "Cotter"]
    with pytest.raises(ImportError, match="tkinter not available.*cannot create GUI"):
        guitools.MultiColumnListbox("Select person", headers, [lst1, lst2, lst3])
