from __future__ import annotations
import pandas as pd
from eda_cli.core import (
    compute_quality_flags,
    correlation_matrix,
    flatten_summary_for_print,
    missing_table,
    summarize_dataset,
    top_categories,
    DatasetSummary,
)


# Вспомогательная функция для создания простого DataFrame
def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [10, 20, 30, None],
            "height": [140, 150, 160, 170],
            "city": ["A", "B", "A", None],
        }
    )


def test_summarize_dataset_basic():
    df = _sample_df()
    summary = summarize_dataset(df)

    assert summary.n_rows == 4
    assert summary.n_cols == 3
    assert any(c.name == "age" for c in summary.columns)
    assert any(c.name == "city" for c in summary.columns)

    summary_df = flatten_summary_for_print(summary)
    assert "name" in summary_df.columns
    assert "missing_share" in summary_df.columns


def test_missing_table_and_quality_flags():
    df = _sample_df()
    missing_df = missing_table(df)

    assert "missing_count" in missing_df.columns
    assert missing_df.loc["age", "missing_count"] == 1

    summary = summarize_dataset(df)
    flags = compute_quality_flags(df, missing_df)  # Передаем df
    assert 0.0 <= flags["quality_score"] <= 1.0


def test_correlation_and_top_categories():
    df = _sample_df()
    corr = correlation_matrix(df)
    # корреляция между age и height существует
    assert "age" in corr.columns or corr.empty is False

    top_cats = top_categories(df, max_columns=5, top_k=2)
    assert "city" in top_cats
    city_table = top_cats["city"]
    assert "value" in city_table.columns
    assert len(city_table) <= 2


# Пример DataFrame с константной колонкой
def _sample_df_with_constant_column() -> pd.DataFrame:
    return pd.DataFrame({
        "age": [10, 20, 30, 40],
        "constant_column": [1, 1, 1, 1],  # Константная колонка
        "height": [140, 150, 160, 170]
    })


def test_has_constant_columns():
    df = _sample_df_with_constant_column()  # Создаем DataFrame с константной колонкой
    summary = summarize_dataset(df)  # Получаем сводку по данным
    missing_df = missing_table(df)  # Создаем таблицу пропусков (не используется в этой проверке)

    flags = compute_quality_flags(df, missing_df)  # Передаем df
    assert flags["has_constant_columns"] == True  # Проверяем, что флаг установился в True для константной колонки


# Пример DataFrame с категорией с множеством уникальных значений
def _sample_df_with_high_cardinality() -> pd.DataFrame:
    return pd.DataFrame({
        "age": [10, 20, 30, 40],
        "city": ["A", "B", "C", "D"],  # Категориальный столбец с 4 уникальными значениями
        "height": [140, 150, 160, 170]
    })


def test_has_high_cardinality_categoricals():
    df = _sample_df_with_high_cardinality()  # Создаем DataFrame с категорией с большим количеством уникальных значений
    summary = summarize_dataset(df)  # Получаем сводку по данным
    missing_df = missing_table(df)  # Создаем таблицу пропусков (не используется в этой проверке)

    flags = compute_quality_flags(df, missing_df)  # Передаем df
    assert flags["has_high_cardinality"] == True  # Проверяем, что флаг установился в True для высокой кардинальности


# Пример DataFrame с дублирующимися идентификаторами
def _sample_df_with_duplicates() -> pd.DataFrame:
    return pd.DataFrame({
        "user_id": [1, 2, 3, 2],  # Дублирующийся user_id
        "age": [10, 20, 30, 40],
        "height": [140, 150, 160, 170]
    })


def test_has_suspicious_id_duplicates():
    df = _sample_df_with_duplicates()  # Создаем DataFrame с дублирующимися идентификаторами
    summary = summarize_dataset(df)  # Получаем сводку по данным
    missing_df = missing_table(df)  # Создаем таблицу пропусков (не используется в этой проверке)

    flags = compute_quality_flags(df, missing_df)  # Передаем df
    assert flags["has_suspicious_id_duplicates"] == True  # Проверяем, что флаг установился в True для дублирующихся идентификаторов
