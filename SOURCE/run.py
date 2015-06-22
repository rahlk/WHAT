#! /Users/rkrsn/miniconda/bin/python
from __future__ import print_function, division

from os import environ, getcwd
from os import walk
from pdb import set_trace
import sys
from bdb import set_trace

# Update PYTHONPATH
HOME = environ['HOME']
axe = HOME + '/git/axe/axe/'  # AXE
pystat = HOME + '/git/pystats/'  # PySTAT
cwd = getcwd()  # Current Directory
sys.path.extend([axe, pystat, cwd])

from Planning import *
from Prediction import *
from WHAT import treatments as treatments2
from _imports import *
from abcd import _Abcd
from cliffsDelta import cliffs
from contrastset import *
from dEvol import tuner
from dectree import *
from demos import cmd
from methods1 import *
from sk import rdivDemo
import numpy as np
import pandas as pd
import csv


class run():

  def __init__(
          self,
          pred=rforest,
          _smoteit=True,
          _n=-1,
          _tuneit=False,
          dataName='ant',
          reps=1,
          extent=0.75,
          fSelect=True,
          Prune=False,
          infoPrune=0.75):
    self.pred = pred
    self.extent = extent
    self.fSelect = fSelect
    self.Prune = Prune
    self.infoPrune = infoPrune
    self.dataName = dataName
    self.out, self.out_pred = [], []
    self._smoteit = _smoteit
    self.train, self.test = self.categorize()
    self.reps = reps
    self._n = _n
    self.tunedParams = None if not _tuneit else tuner(
        self.pred, self.train[_n])
    self.headers = createTbl(
        self.train[
            self._n],
        isBin=True,
        bugThres=1).headers

  def categorize(self):
    dir = '../Data'
    self.projects = [Name for _, Name, __ in walk(dir)][0]
    self.numData = len(self.projects)  # Number of data
    one, two = explore(dir)
    data = [one[i] + two[i] for i in xrange(len(one))]

    def withinClass(data):
      N = len(data)
      return [(data[:n], [data[n]]) for n in range(1, N)]

    def whereis():
      for indx, name in enumerate(self.projects):
        if name == self.dataName:
          return indx

    return [
        dat[0] for dat in withinClass(data[whereis()])], [
        dat[1] for dat in withinClass(data[whereis()])]  # Train, Test

  def go(self):

    for _ in xrange(self.reps):
      predRows = []
      train_DF = createTbl(self.train[self._n], isBin=True, bugThres=1)
      test_df = createTbl(self.test[self._n], isBin=True, bugThres=1)
      actual = Bugs(test_df)
      before = self.pred(train_DF, test_df,
                         tunings=self.tunedParams,
                         smoteit=True)

      for predicted, row in zip(before, test_df._rows):
        tmp = row.cells
        tmp[-2] = predicted
        if predicted > 0:
          predRows.append(tmp)

      predTest = clone(test_df, rows=predRows)

      if predRows:
        newTab = treatments2(
            train=self.train[self._n],
            test=self.test[self._n],
            test_df=predTest,
            extent=self.extent,
            far=False,
            infoPrune=self.infoPrune,
            Prune=self.Prune).main()
      else:
        newTab = treatments2(
            train=self.train[
                self._n],
            test=self.test[
                self._n],
            far=False,
            extent=self.extent,
            infoPrune=self.infoPrune,
            Prune=self.Prune).main()

      after = self.pred(train_DF, newTab,
                        tunings=self.tunedParams,
                        smoteit=True)

#       set_trace()
      self.out_pred.append(_Abcd(before=actual, after=before))
      delta = cliffs(lst2=Bugs(predTest), lst1=after).delta()
      self.out.append(delta)
    if self.extent == 0:
      append = 'Baseline'
    else:
      if self.Prune:
        append = str(
            self.extent) + ', iP= ' + str(
            int(self.infoPrune * 100)) + r'\%' if not self.fSelect else str(
            self.extent) + ', weight, iP= ' + str(
            int(self.infoPrune * 100)) + r'\%'
      else:
        append = str(
            self.extent) if not self.fSelect else str(
            self.extent) + ', weight'

    self.out.insert(0, self.dataName + ', ' + append)
    self.out_pred.insert(0, self.dataName)
    print(self.out)

  def deltas(self, whatnow='deltas'):

    predRows = []
    delta = []
    train_DF = createTbl(self.train[self._n], isBin=True, bugThres=1)
    test_df = createTbl(self.test[self._n], isBin=True, bugThres=1)
    actual = Bugs(test_df)
    before = self.pred(train_DF, test_df,
                       tunings=self.tunedParams,
                       smoteit=True)

    allRows = [r.cells[:-2] for r in train_DF._rows + test_df._rows]
    for predicted, row in zip(before, test_df._rows):
      tmp = row.cells
      tmp[-2] = predicted
      if predicted > 0:
        predRows.append(tmp)

    predTest = clone(test_df, rows=predRows)

    if predRows:
      newTab = treatments2(
          train=self.train[self._n],
          test=self.test[self._n],
          test_df=predTest,
          extent=self.extent,
          far=False,
          infoPrune=self.infoPrune,
          Prune=self.Prune).deltas()
    else:
      newTab = treatments2(
          train=self.train[
              self._n],
          test=self.test[
              self._n],
          far=False,
          extent=self.extent,
          infoPrune=self.infoPrune,
          Prune=self.Prune).deltas()

    def min_max():
      N = len(allRows[0])
      base = lambda X: sorted(X)[-1] - sorted(X)[0]
      return [base([r[i] for r in allRows]) for i in xrange(N)]

    for newRow in newTab:
      delta.append((np.array(newRow) / np.array(min_max())))

    if whatnow == 'deltas':
      return delta
    if whatnow == 'minmax':
      return allRows


def _test(file):
  """
  Baselining
  """
  R = run(
      dataName=file,
      extent=0.00,
      reps=24,
      fSelect=False,
      Prune=False,
      infoPrune=None).go()

  """
  Mutation without Feature Selection
  """
  R = run(
      dataName=file,
      extent=0.25,
      reps=24,
      fSelect=False,
      Prune=False,
      infoPrune=None).go()

  R = run(
      dataName=file,
      extent=0.50,
      reps=24,
      fSelect=False,
      Prune=False,
      infoPrune=None).go()

  R = run(
      dataName=file,
      extent=0.75,
      reps=24,
      fSelect=False,
      Prune=False,
      infoPrune=None).go()

  """
  Mutation with Feature Selection
  """
  R = run(
      dataName=file,
      extent=0.25,
      reps=24,
      fSelect=True,
      Prune=False,
      infoPrune=None).go()
  R = run(
      dataName=file,
      extent=0.50,
      reps=24,
      fSelect=True,
      Prune=False,
      infoPrune=None).go()
  R = run(
      dataName=file,
      extent=0.75,
      reps=24,
      fSelect=True,
      Prune=False,
      infoPrune=None).go()

  """
  Information Pruning with Feature Selection
  """
  R = run(
      dataName=file,
      extent=0.25,
      reps=24,
      fSelect=True,
      Prune=True,
      infoPrune=0.5).go()

  R = run(
      dataName=file,
      extent=0.50,
      reps=24,
      fSelect=True,
      Prune=True,
      infoPrune=0.50).go()

  R = run(
      dataName=file,
      extent=0.75,
      reps=24,
      fSelect=True,
      Prune=True,
      infoPrune=0.5).go()


def deltaCSVwriter():
  for name in ['ant', 'ivy', 'jedit', 'lucene', 'poi']:
    delta = run(
        dataName=name,
        extent=0.75,
        reps=24,
        fSelect=True,
        Prune=False,
        infoPrune=0.5).deltas()

    y = np.median(delta, axis=0)
    yhi, ylo = np.percentile(delta, q=[90, 10], axis=0)
    dat1 = sorted([(h.name[1:], a, b, c) for h, a, b, c in zip(
        run().headers[:-2], y, ylo, yhi)], key=lambda F: F[1])
    dat = np.asarray([(d[0], n, d[1], d[2], d[3])
                      for d, n in zip(dat1, range(1, 21))])
    with open('/Users/rkrsn/git/GNU-Plots/rkrsn/errorbar/%s.csv' % (name), 'w') as csvfile:
      writer = csv.writer(csvfile, delimiter=' ')
      for el in dat[()]:
        writer.writerow(el)


def minmaxCSVwriter():
  minMax = []
  for name in ['ant', 'ivy', 'jedit', 'lucene', 'poi']:
    minMax.extend(run(
        dataName=name,
        extent=0.75,
        reps=24,
        fSelect=True,
        Prune=False,
        infoPrune=0.5).deltas(whatnow='minmax'))
  dat = np.asarray([(h.name[1:],
                     min([r[i] for r in minMax]),
                     max([r[i] for r in minMax])) for i,
                    h in enumerate(run().headers[:-2])])
  with open('/Users/rkrsn/git/WHAT/SOURCE/minmax/%s.csv' % ('minmax'), 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=' ')
    writer.writerow(['Attr', 'Min', 'Max'])
    for el in dat[()]:
      writer.writerow(el)


if __name__ == '__main__':
  deltaCSVwriter()
#   minmaxCSVwriter()
#   eval(cmd())
