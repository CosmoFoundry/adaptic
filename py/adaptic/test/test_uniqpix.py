from pathlib import Path
import tempfile
import unittest

import numpy as np

from .util import make_uniqpix_table
from adaptic.desi import find_and_concat_uniqpix_tables


class TestFindAndConcatUniqpixTables(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.datadir = Path(self.tempdir.name)

        # Three (survey, program) combinations with non-overlapping UNIQPIX ranges
        # so each test can confirm exactly which files were included or excluded.
        #   A: main/dark    -> UNIQPIX 100, 200
        #   B: sv1/bright   -> UNIQPIX 300, 400
        #   C: sv3/other    -> UNIQPIX 500, 600
        #
        # Ignoring survey='main' removes A; ignoring program='bright' removes B;
        # ignoring both still leaves C, which is what test_ignore_both verifies.
        self.file_a = make_uniqpix_table(
            self.datadir, 'main', 'dark', [100, 200], [10, 20])
        self.file_b = make_uniqpix_table(
            self.datadir, 'sv1', 'bright', [300, 400], [30, 40])
        self.file_c = make_uniqpix_table(
            self.datadir, 'sv3', 'other', [500, 600], [50, 60])

        self.all_pairs = {('main', 'dark'), ('sv1', 'bright'), ('sv3', 'other')}

    def _pairs(self, result):
        return {(row['SURVEY'], row['PROGRAM']) for row in result}

    def test_no_ignores(self):
        result = find_and_concat_uniqpix_tables(self.datadir)
        self.assertEqual(self._pairs(result), self.all_pairs)
        self.assertEqual(set(result['UNIQPIX']), {100, 200, 300, 400, 500, 600})

    def test_ignore_survey(self):
        result = find_and_concat_uniqpix_tables(self.datadir, ignore_survey=['main'])
        self.assertEqual(self._pairs(result), {('sv1', 'bright'), ('sv3', 'other')})
        self.assertNotIn('main', result['SURVEY'])
        self.assertEqual(set(result['UNIQPIX']), {300, 400, 500, 600})

    def test_ignore_program(self):
        result = find_and_concat_uniqpix_tables(self.datadir, ignore_program=['bright'])
        self.assertEqual(self._pairs(result), {('main', 'dark'), ('sv3', 'other')})
        self.assertNotIn('bright', result['PROGRAM'])
        self.assertEqual(set(result['UNIQPIX']), {100, 200, 500, 600})

    def test_ignore_both(self):
        result = find_and_concat_uniqpix_tables(
            self.datadir, ignore_survey=['main'], ignore_program=['bright'])
        self.assertEqual(self._pairs(result), {('sv3', 'other')})
        self.assertEqual(set(result['UNIQPIX']), {500, 600})

    def tearDown(self):
        self.tempdir.cleanup()


if __name__ == '__main__':
    unittest.main()
