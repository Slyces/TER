# -*- coding: utf-8 -*-
# Generated by Django 1.10.5 on 2017-04-02 14:59
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('hello_world', '0002_auto_20170401_1730'),
    ]

    operations = [
        migrations.CreateModel(
            name='PredictedIndexes',
            fields=[
                ('id', models.IntegerField(blank=True, db_column='Id', primary_key=True, serialize=False)),
                ('label', models.CharField(db_column='Label', max_length=20)),
                ('datetime', models.DateTimeField(db_column='DateTime')),
                ('nasqad', models.DecimalField(db_column='Nasdaq', decimal_places=19, max_digits=20)),
                ('dowjones', models.DecimalField(db_column='Dowjones', decimal_places=19, max_digits=20)),
                ('snp500', models.DecimalField(db_column='SnP500', decimal_places=19, max_digits=20)),
                ('rates', models.DecimalField(db_column='Rates', decimal_places=19, max_digits=20)),
            ],
            options={
                'db_table': 'predicted_indexes',
            },
        ),
        migrations.AlterModelTable(
            name='historicindexes',
            table='historic_indexes',
        ),
    ]
