#pragma once
#include <vector>
#include <queue>
#include <functional>
#include "manager/system_description.h"
namespace ecs
{
  template<typename E>
  struct EventDescription;
  template<typename E>
  struct SingleEventDescription;
  struct EntityContainer;
  struct Core
  {
    std::unordered_map<uint, FullTypeDescription> types;
    std::vector<SystemDescription*> systems;
    std::vector<QueryDescription*> queries;
    std::vector<QueryDescription*> event_queries;
    EntityContainer *entityContainer;
    std::queue<std::function<void()>> events;
    std::queue<EntityId> toDestroy;
    uint currentSceneTags;
    Core();
    ~Core();
    template<typename E>
    std::vector<EventDescription<E>*> &events_handler()
    {
      static std::vector<EventDescription<E>*> handlers;
      return handlers;
    }
    template<typename E>
    std::vector<SingleEventDescription<E>*> &single_events_handler()
    {
      static std::vector<SingleEventDescription<E>*> handlers;
      return handlers;
    }
    void destroy_all_entities();
    void destroy_entities_from_destroy_queue(bool with_swap_last_element);
    void replace_entity_container(EntityContainer *container);
    static bool allow_system_execute(uint tags, uint require_tags);
  };
  Core &core();

  template<typename E>
  struct EventDescription : QueryDescription
  {
    typedef  void (*EventHandler)(const E&);
    EventHandler eventHandler;
    uint tags;
    EventDescription(const char *name, const std::vector<FunctionArgument> &args, EventHandler eventHandler, uint tags):
      QueryDescription(name, args, false), eventHandler(eventHandler), tags(tags)
    {
      core().events_handler<E>().push_back(this);
      core().event_queries.push_back((QueryDescription*)this);
      //printf("EventDescription %s\n", name);
    }
  };
  template<typename E>
  struct SingleEventDescription : QueryDescription
  {
    typedef  void (*EventHandler)(const E&, ecs::QueryIterator&);
    EventHandler eventHandler;
    uint tags;
    SingleEventDescription(const char *name, const std::vector<FunctionArgument> &args, EventHandler eventHandler, uint tags):
      QueryDescription(name, args, false), eventHandler(eventHandler), tags(tags)
    {
      core().single_events_handler<E>().push_back(this);
      core().event_queries.push_back((QueryDescription*)this);
    }
  };


  template<typename T>
  T* get_component(const EntityId &entity, const char *name);

  template<typename T>
  struct ComponentInitializer
  {
    const char *name;
    const T & component;
  };
  template<typename T>
  string_hash component_initializer_hash(const ComponentInitializer<T> &component)
  {
    return get_type_description<T>(component.name).type_name_hash();
  }
  template<typename T> 
  void component_copy(void *data, const T & component)
  {
    new (data) T(std::move(component));
  }
  template<int N, typename T, typename ...Args>
  void component_initializer_copy(vector<void*> &data, const ComponentInitializer<T> & arg, const Args &...args)
  {
    component_copy(data[N], arg.component);
    if constexpr (sizeof...(args) > 0)
      component_initializer_copy<N+1>(data, args...);
  }
  bool check_entity_container();
  pair<EntityId, Archetype&> add_entity(const vector<string_hash> & type_hashes);
  template<typename ...Args>
  EntityId create_entity(const ComponentInitializer<Args> &...args)
  {
    if (check_entity_container())
      return EntityId();
    vector<string_hash> typeHashes = 
      {get_type_description<EntityId>("eid").type_name_hash(), component_initializer_hash(args)... };
    auto [eid, archetype] = add_entity(typeHashes);
    vector<void*> data = archetype.get_entity_data(typeHashes);
    component_copy(data[0], eid);
    component_initializer_copy<1>(data, args...);
    send_event_immediate(eid, OnEntityCreated());
    return eid;
  }
  EntityId create_entity(const string &template_name);
  EntityId create_entity(const char *template_name);
  struct Template;
  EntityId create_entity(const Template *temp);
  EntityId find_entity(uint archetype, uint index);
  void destroy_entity(const EntityId &eid);



  template<typename E>
  void send_event(const E &event)
  {
    core().events.push([event](){
      for (EventDescription<E> *descr : core().events_handler<E>())
        if (Core::allow_system_execute(descr->tags, core().currentSceneTags))
          descr->eventHandler(event);
    });
  }
  template<typename E>
  void send_event_immediate(const E &event)
  {
    for (EventDescription<E> *descr : core().events_handler<E>())
      if (Core::allow_system_execute(descr->tags, core().currentSceneTags))
        descr->eventHandler(event);
  }


  bool get_iterator(const QueryDescription &descr, const EntityId &eid, QueryIterator &iterator);
  bool get_iterator(const SingleQueryDescription &descr, const EntityId &eid, QueryIterator &iterator);


  template<typename E>
  void send_event(const EntityId &eid, const E &event)
  {
    if (eid)
    {
      core().events.push([eid, event](){
        for (SingleEventDescription<E> *descr : core().single_events_handler<E>())
        {
          if (Core::allow_system_execute(descr->tags, core().currentSceneTags))
          {
            QueryIterator iterator;
            if (get_iterator(*((QueryDescription*)descr), eid, iterator))
            {
              descr->eventHandler(event, iterator);
            }
          }
        } 
      });
    }
  }
  template<typename E>
  void send_event_immediate(const EntityId &eid, const E &event)
  {
    if (eid)
    {
      for (SingleEventDescription<E> *descr : core().single_events_handler<E>())
      {
        if (Core::allow_system_execute(descr->tags, core().currentSceneTags))
        {
          QueryIterator iterator;
          if (get_iterator(*((QueryDescription*)descr), eid, iterator))
          {
            descr->eventHandler(event, iterator);
          }
        }
      } 
    }
  }

  void system_statistic();
  void destroy_scene();
}
