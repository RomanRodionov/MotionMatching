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
    void destroy_entities();
    void replace_entity_container(EntityContainer *container);
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
      printf("EventDescription %s\n", name);
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

  EntityId create_entity(ComponentInitializerList &list);
  EntityId create_entity(const std::string &template_name);
  EntityId create_entity(const char *template_name);
  struct Template;
  EntityId create_entity(const Template *temp);
  void destroy_entity(const EntityId &eid);



  template<typename E>
  void send_event(const E &event)
  {
    uint currentSceneTags = core().currentSceneTags;
    core().events.push([event, currentSceneTags](){
    for (EventDescription<E> *descr : core().events_handler<E>())
      if (descr->tags & currentSceneTags)
        descr->eventHandler(event);
    });
  }


  bool get_iterator(const QueryDescription &descr, const EntityId &eid, QueryIterator &iterator);
  bool get_iterator(const SingleQueryDescription &descr, const EntityId &eid, QueryIterator &iterator);


  template<typename E>
  void send_event(const EntityId &eid, const E &event)
  {
    if (eid)
    {
      uint currentSceneTags = core().currentSceneTags;
      core().events.push([eid, event, currentSceneTags](){
        for (SingleEventDescription<E> *descr : core().single_events_handler<E>())
        {
          if (descr->tags & currentSceneTags)
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
      uint currentSceneTags = core().currentSceneTags;
      for (SingleEventDescription<E> *descr : core().single_events_handler<E>())
      {
        if (descr->tags & currentSceneTags)
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
  template<typename E>
  void send_event_immediate(const E &event)
  {
    uint currentSceneTags = core().currentSceneTags;
    for (EventDescription<E> *descr : core().events_handler<E>())
      if (descr->tags & currentSceneTags)
        descr->eventHandler(event);
  }

  void system_statistic();

}

  typedef uint32_t uint;