'''
XML文件接卸
最初用于提取valence和arousal的值
'''


import xml.dom.minidom


def read_xml(filepath):
    dom = xml.dom.minidom.parse(filepath)
    session = dom.documentElement
    arousal = float(session.getAttribute("feltArsl"))
    print("arousal: ", arousal, "  (type: ", type(arousal), ")")
    valence = session.getAttribute("feltVlnc")
    print("valence: ", valence, "  (type: ", type(valence), ")")


def test(filepath):
    dom = xml.dom.minidom.parse(filepath)
    print('read xml successfully.')
    root = dom.documentElement

    itemList = root.getElementsByTagName('login')
    item = itemList[0]

    un = item.getAttribute("username")
    print(un)

    pd = item.getAttribute("passwd")
    print(pd)

    ii = root.getElementsByTagName('item')
    i1 = ii[0]
    i = i1.getAttribute("id")
    print(i)

    i2 = ii[1]
    i = i2.getAttribute("id")
    print(i)


if __name__ == '__main__':
    read_xml('session.xml')
    # test(filepath='abc.xml')
