import streamlit as st
import pandas as pd
from geopy.distance import geodesic
import folium
from streamlit_folium import st_folium
from io import BytesIO
import networkx as nx
import urllib.parse
import re
import requests
import json

def total_distance(route, coords):
    return sum(geodesic(coords[route[i]], coords[route[i+1]]).meters for i in range(len(route)-1)) + geodesic(coords[route[-1]], coords[route[0]]).meters

def two_opt(route, coords, max_iter=1000):
    best = route.copy()
    improved = True
    iter_count = 0
    while improved and iter_count < max_iter:
        improved = False
        for i in range(1, len(route) - 2):
            for j in range(i + 1, len(route)):
                if j - i == 1: continue
                new_route = best[:i] + best[i:j][::-1] + best[j:]
                if total_distance(new_route, coords) < total_distance(best, coords):
                    best = new_route
                    improved = True
        iter_count += 1
    return best

def nearest_neighbor_route(coords):
    visited = [0]
    while len(visited) < len(coords):
        last = visited[-1]
        next_idx = min(
            (i for i in range(len(coords)) if i not in visited),
            key=lambda i: geodesic(coords[last], coords[i]).meters
        )
        visited.append(next_idx)
    return visited

def create_distance_graph(coords):
    G = nx.Graph()
    n = len(coords)
    for i in range(n):
        for j in range(i+1, n):
            dist = geodesic(coords[i], coords[j]).meters
            G.add_edge(i, j, weight=dist)
    return G

def christofides_tsp(coords):
    G = create_distance_graph(coords)
    cycle = nx.approximation.traveling_salesman_problem(
        G, cycle=True, method=nx.approximation.christofides
    )
    return cycle

st.set_page_config(layout="wide")

st.title("TSP 최적 경로 계산기 (알고리즘 선택, 지도 시각화 포함)")

col1, col2 = st.columns([1, 2])

with col1:
    algorithm = st.selectbox(
        "사용할 알고리즘을 선택하세요",
        ("최근접 이웃(Nearest Neighbor)", "2-opt", "Christofides"),
        index=2
    )
    data_source = st.radio("데이터 소스 선택", ("엑셀 업로드", "구글 시트 URL"), horizontal=True)
    uploaded_file = None
    sheet_url = None
    if data_source == "엑셀 업로드":
        uploaded_file = st.file_uploader("엑셀 파일을 업로드하세요", type=["xlsx"])
    else:
        sheet_url = st.text_input("구글 시트 URL을 입력하세요 (공개 시트)")
        gid = st.text_input("시트 GID를 입력하세요 (주소창 gid=... 값)")

with col2:
    map_placeholder = st.empty()

# 데이터프레임 불러오기
if data_source == "엑셀 업로드" and uploaded_file:
    df = pd.read_excel(uploaded_file)
elif data_source == "구글 시트 URL" and sheet_url and gid:
    match = re.match(r"https://docs.google.com/spreadsheets/d/([a-zA-Z0-9-_]+)", sheet_url)
    if match:
        sheet_id = match.group(1)
        csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"
        try:
            df = pd.read_csv(csv_url)
        except Exception as e:
            st.error(f"구글 시트에서 데이터를 불러올 수 없습니다: {e}")
            df = None
    else:
        st.error("유효한 구글 시트 URL을 입력하세요.")
        df = None
else:
    df = None

if df is not None:
    # '폐업여부' 컬럼이 있으면 Y가 아닌 데이터만 사용
    if '폐업여부' in df.columns:
        df = df[df['폐업여부'] != 'Y']
    if not all(col in df.columns for col in ['상호명', '도로명주소', '위도', '경도']):
        st.error("데이터에 '상호명', '도로명주소', '위도', '경도' 컬럼이 필요합니다.")
        st.stop()

    locations = df[['상호명', '도로명주소', '위도', '경도']].dropna().reset_index(drop=True)
    coords = locations[['위도', '경도']].values.tolist()

    if algorithm == "최근접 이웃(Nearest Neighbor)":
        route_order = nearest_neighbor_route(coords)
        file_suffix = "nearest"
    elif algorithm == "2-opt":
        initial_route = nearest_neighbor_route(coords)
        route_order = two_opt(initial_route, coords)
        file_suffix = "2opt"
    elif algorithm == "Christofides":
        route_order = christofides_tsp(coords)
        file_suffix = "christofides"
    else:
        st.error("알고리즘 선택 오류")
        st.stop()

    ordered_df = locations.iloc[route_order].copy()
    ordered_df.insert(0, '방문 순서', list(range(1, len(route_order)+1)))

    # 지도 시각화 (순환 경로)
    m = folium.Map(location=coords[route_order[0]], zoom_start=15)
    folium.Marker(coords[route_order[0]], popup="출발지", icon=folium.Icon(color='red')).add_to(m)
    for i, idx in enumerate(route_order):
        lat, lon = coords[idx]
        name = locations.iloc[idx]['상호명']
        tmap_url = f"tmap://route?goalx={lon}&goaly={lat}&goalname={urllib.parse.quote(str(name))}"
        kakao_url = f"https://map.kakao.com/link/to/{urllib.parse.quote(str(name))},{lat},{lon}"
        popup_html = f"""
            <b>{i+1} - {name}</b><br>
            <a href='{tmap_url}' target='_blank'>경로안내(티맵)</a><br>
            <a href='{kakao_url}' target='_blank'>경로안내(카카오맵)</a>
        """
        folium.Marker(
            [lat, lon],
            popup=folium.Popup(popup_html, max_width=300),
            icon=folium.Icon(color='blue' if i > 0 else 'red')
        ).add_to(m)
    # 경로 polyline (시작점으로 복귀)
    route_coords = [coords[i] for i in route_order]
    route_coords.append(coords[route_order[0]])
    folium.PolyLine(route_coords, color="blue", weight=2.5, opacity=1).add_to(m)
    with col2:
        st_folium(m, width=1600, height=900)

    # 결과 엑셀 다운로드
    output = BytesIO()
    result_df = df.copy()
    result_df.insert(0, '방문 순서', "")
    for i, idx in enumerate(route_order):
        result_df.at[idx, '방문 순서'] = i+1
    with col1:
        result_df.to_excel(output, index=False)
        st.download_button(
            label="최적 경로 엑셀 다운로드",
            data=output.getvalue(),
            file_name=f"최적경로결과_{file_suffix}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        ) 