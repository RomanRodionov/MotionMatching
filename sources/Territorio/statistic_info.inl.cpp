#include "statistic_info.inl"
//Code-generator production

ecs::QueryDescription gather_lands_descr("gather_lands", {
  {ecs::get_type_description<uint>("landIndex"), false},
  {ecs::get_type_description<int>("landCount"), false},
  {ecs::get_type_description<uint>("forces"), false},
  {ecs::get_type_description<ecs::Tag>("isPlayableLand"), false}
}, {
});

template<typename Callable>
void gather_lands(Callable lambda)
{
  ecs::perform_query<uint, int, uint>
  (gather_lands_descr, lambda);
}


void show_statistic_func();

ecs::SystemDescription show_statistic_descr("show_statistic", {
  {ecs::get_type_description<vector<vec3>>("land_colors"), false}
}, {
}, {},
show_statistic_func, ecs::stage::ui, ecs::tags::all,
{},
{});

void show_statistic_func()
{
  ecs::perform_system(show_statistic_descr, show_statistic);
}


