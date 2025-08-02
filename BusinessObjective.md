# Project: HDB Resale Flat Interactive Visualization Tool

## 1. Problem Statement

Prospective HDB flat buyers face a complex and data-rich market. They often lack an intuitive, easy-to-use tool to explore available options, understand market trends, and find properties that match their specific budget, location, and lifestyle needs. This makes the decision-making process time-consuming and overwhelming.

## 2. Proposed Solution

To address this, we will develop an interactive visualization tool that leverages the provided HDB resale dataset. The tool will empower new house buyers by providing data-driven insights through user-friendly maps, charts, and filters, thereby simplifying their property search and decision-making process.

## Datasets
*   `datasets/train.csv`: Contains historical HDB resale transaction data along with flat attributes and proximity to various amenities.

## 3. Target Audience
*   Prospective HDB resale flat buyers, particularly first-time buyers and families, who need to understand market trends and find properties that fit their specific criteria (e.g., budget, family size, proximity to schools or workplaces).

## 4. Core Features

1. **Dynamic Filtering**: Filter selection (if selection is left blank, use all data):
    * **Budget Range** - Allow users to select a budget range using a slider.
    * **Location** - Filter by `town`
    * **Flat Type** - Filter by `flat_type` (e.g., 3 ROOM, 4 ROOM).
    * **Floor Area** - Filter by `floor_area_sqm`.
    * **Lease Commencement Date** - Filter by `lease_commence_date` or `hdb_age` (if available).
    * **Amenities** - Filter by proximity to amenities like MRT stations, malls, and schools (if columns exist) as checkboxes.

2. **Interactive Map Visualization**:
    * Display properties on a map of Singapore using `Latitude` and `Longitude`.
    * Use visual cues (e.g., color, size) for markers to represent `resale_price`.
    * Display key information on hover/click (e.g., price, address, floor area).
    * Allow toggling overlays for key amenities like MRT stations (`mrt_name`), primary schools (`pri_sch_name`), and malls.

3. **Search Results Table**:
    * Display a table of properties matching the selected filters.
    * Include key attributes like `resale_price`, `town`, `flat_type`, `floor_area_sqm`, and `lease_commence_date`.
    * Allow sorting by any column and searching by address or flat type.

4. **Interesting Facts**:
    * Display interesting facts about the dataset, such as:
        * Average resale price by town.
        * Most common flat types.
        * Trends in resale prices over time.
    * Use charts to visualize these facts (e.g., bar charts, line graphs).
5. **Price Trends**:
    * Show a line chart of median resale prices over time (`Tranc_YearMonth`).
    * Allow users to see how prices have changed in their selected town or flat type.

6. **Property Comparison Tool**: Enable users to select 2-3 properties and view their key attributes in a side-by-side comparison table.

7. **Preference-Based Recommender**: Allow users to input their preferences (e.g., desired town, flat type, budget) to receive a ranked list of suitable properties from the dataset.

## 8. Success Metrics
*   **Task Completion**: A user can successfully filter, find, and compare at least three properties that match their criteria.
*   **User Engagement**: High interaction rates with filters, charts, and the map view.
*   **User Satisfaction**: Positive feedback collected via a simple survey regarding the tool's usability and usefulness in their home-buying research.