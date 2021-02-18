#! encoding = utf-8

""" Search OCS frequency """


import sqlite3


def decode_cat(line):
    """ Decode the line in SPFIT format
    :argument
        line: str           a line in .CAT format
    :returns
        cat: tuple          a catalog tuple
            freq: float
            unc: float
            inten: float
            elow: float
            eup: float
            qns: list of int
    """

    freq = float(line[0:13])
    unc = float(line[13:21])
    inten = 10**(float(line[21:29]))
    elow = float(line[31:41])
    gup = int(line[41:44])
    qn_str = line[55:67].rstrip()
    n = len(qn_str)
    qns_up = list(int(qn_str[idx:idx+2]) for idx in range(0, n, 2))
    qn_str = line[67:79].rstrip()
    n = len(qn_str)
    qns_low = list(int(qn_str[idx:idx + 2]) for idx in range(0, n, 2))
    eup = elow + freq / 2.99792458e4

    return (freq, unc, inten, elow, eup, gup, *qns_up, *qns_low)


def create_db(catalogs):
    """ Create OCS DB with catalogs """

    conn = sqlite3.connect('sample_data/OCS/OCS_freq.db')
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS ocs (
              freq DOUBLE, 
              species TEXT,
              mass INTEGER, 
              J2 INTEGER,
              J1 INTEGER);""")
    conn.commit()

    sql = "INSERT into ocs (freq, species, mass, J2, J1) VALUES (?,?,?,?,?)"
    for species, value in catalogs.items():
        mass, filename = value
        with open(filename, 'r') as f:
            for a_line in f:
                item = decode_cat(a_line)
                freq = item[0]
                j2 = item[6]
                j1 = item[7]
                c.execute(sql, (freq, species, mass, j2, j1))
        conn.commit()
    conn.close()


def search_freq(c, fmin, fmax):
    """ Search frequency in this frequency range """

    sql = "SELECT * from ocs WHERE freq >= {:g} AND freq <= {:g}".format(fmin, fmax)
    c.execute(sql)
    print('freq', 'species', 'mass', 'J2', 'J1')
    for item in c.fetchall():
        print(*item)


def search_freq_txt(c, txt):
    """ Search frequency in this frequency range """

    _l = txt.split('|')
    fmin = int(_l[0].split('=')[1])
    fmax = int(_l[1].split('=')[1])

    sql = "SELECT species, mass from ocs WHERE freq >= {:g} AND freq <= {:g}".format(fmin, fmax)
    c.execute(sql)
    # print('freq', 'species', 'mass', 'J2', 'J1')
    for item in c.fetchall():
        print('|species={:s}|mass={:d}'.format(*item))


if __name__ == '__main__':

    create_db({
                'OCS': (60, 'sample_data/Catalogs/OCS_v=0.cat'),
                '18OCS': (62, 'sample_data/Catalogs/18OCS.cat'),
                #'18O13CS': (63, 'sample_data/Catalogs/18O13CS.cat'),
                #'18OC34S': (64, 'sample_data/Catalogs/18OC34S.cat'),
                'O13CS': (61, 'sample_data/Catalogs/O13CS.cat'),
                #'O13C33S': (62, 'sample_data/Catalogs/O13C33S.cat'),
                #'O13C34S': (63, 'sample_data/Catalogs/O13C34S.cat'),
                'OC33S': (61, 'sample_data/Catalogs/OC33S.cat'),
                'OC34S': (62, 'sample_data/Catalogs/OC34S.cat'),
                #'OC36S': (64, 'sample_data/Catalogs/OC36S.cat'),
                'v2': (60, 'sample_data/Catalogs/OCS_v2=1.cat'),
    })
