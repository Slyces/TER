from django.db import models
import datetime
from django.utils import timezone


# This is an auto-generated Django model module.
# You'll have to do the following manually to clean this up:
#   * Rearrange models' order
#   * Make sure each model has one field with primary_key=True
#   * Make sure each ForeignKey has `on_delete` set to the desired behavior.
#   * Remove `managed = False` lines if you wish to allow Django to create, modify, and delete the table
# Feel free to rename the models, but don't rename db_table values or field names.


class HistoricIndexes(models.Model):
    id = models.IntegerField(db_column='Id', primary_key=True, blank=True)
    datetime = models.DateTimeField(db_column='DateTime', blank=True, null=True)
    nasqad = models.DecimalField(db_column='Nasdaq', max_digits=20, decimal_places=19)
    dowjones = models.DecimalField(db_column='Dowjones', max_digits=20, decimal_places=19)
    snp500 = models.DecimalField(db_column='SnP500', max_digits=20, decimal_places=19)
    rates = models.DecimalField(db_column='Rates', max_digits=20, decimal_places=19)

    class Meta:
        db_table = 'historic_indexes'


class PredictedIndexes(models.Model):
    id = models.IntegerField(db_column='Id', primary_key=True, blank=True)
    label = models.CharField(db_column="Label", max_length=20)
    datetime = models.DateTimeField(db_column='DateTime')
    nasqad = models.DecimalField(db_column='Nasdaq', max_digits=20, decimal_places=19)
    dowjones = models.DecimalField(db_column='Dowjones', max_digits=20, decimal_places=19)
    snp500 = models.DecimalField(db_column='SnP500', max_digits=20, decimal_places=19)
    rates = models.DecimalField(db_column='Rates', max_digits=20, decimal_places=19)

    class Meta:
        db_table = 'predicted_indexes'
