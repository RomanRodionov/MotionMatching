#include <ecs.h>
#include <application/time.h>
#include <3dmath.h>
#include "constanta.h"


struct DagorTestEntity
{
  mat4 transform;
  int iv = 10, ic2 = 10;
  vec3 p={1,0,0};
  mat4 d[9];
  vec3 v={1,0,0};
  int ivCopy = 10;
  virtual void update(float dt){ p += dt*v;}
  DagorTestEntity(int i) : iv(i), p(rand_vec3()), v(rand_vec3()), ivCopy(i){}
};

struct TinyEntity
{
  vec3 P, V;
  //virtual void update(float dt){ P += dt*V;}
  TinyEntity() : P(rand_vec3()), V(rand_vec3()){}
};

static vector<DagorTestEntity> list0;
static vector<DagorTestEntity*> list1;
static vector<DagorTestEntity*> list2;
static vector<TinyEntity> tinyList0;

static vector<vec3> pData;
static vector<vec3> vData;

volatile int cache0 = 0;
void prune_cache()
{
  return;
  static vector<int> memory;
  if (!memory.size())
    memory.resize(1<<20, 1);
  for (auto i:memory)
    cache0 += i;
}

EVENT(ecs::SystemTag::GameEditor) dag_init(const ecs::OnSceneCreated &)
{

  debug_log("struct sizeof = %d", sizeof(DagorTestEntity));

  {
    TimeScope a("ecs_create");
    for (uint i = 0; i < dagorEntityCount; i++)
    {
      ecs::create_entity<vec3, vec3>(
        {"p", rand_vec3()},
        {"v", rand_vec3()}
      );
    }
  }
  {
    TimeScope b("vector_structs_create");
    for (uint i = 0; i < dagorEntityCount; i++)
    {
      list0.emplace_back(DagorTestEntity(i));
    }
  }
  {
    TimeScope b("vector_pointers_create");
    for (uint i = 0; i < dagorEntityCount; i++)
    {
      list1.emplace_back(new DagorTestEntity(i));
    }
  }
  {
    TimeScope b("vector_pointers_create");
    for (uint i = 0; i < dagorEntityCount; i++)
    {
      list2.emplace_back(new DagorTestEntity(i));
    }
  }
  uint tinyCount = dagorEntityCount*10;
  {
    TimeScope b("tiny_structs_creation");
    for (uint i = 0; i < tinyCount; i++)
      tinyList0.emplace_back();
  }
  {
    TimeScope b("tiny_SoA_creation");
    for (uint i = 0; i < tinyCount; i++)
    {
      pData.emplace_back(rand_vec3());
      vData.emplace_back(rand_vec3());
    }
  }
  {
    TimeScope a("tiny_structs_ecs_creation");
    for (uint i = 0; i < tinyCount; i++)
    {
      ecs::create_entity<vec3, vec3>(
        {"P", rand_vec3()},
        {"V", rand_vec3()}
      );
    }
  }
  fflush(stdout);
  auto t1 = std::thread([&]()
  {
    int a = 1;
    while(1)
    {
      a++;
    };
  });
  
  auto t2 = std::thread([&]()
  {
    int a = 1;
    while(1)
    {
      a++;
    };
  });
  
  auto t3 = std::thread([&]()
  {
    int a = 1;
    while(1)
    {
      a++;
    };
  });
  
  auto t4 = std::thread([&]()
  {
    int a = 1;
    while(1)
    {
      a++;
    };
  });
  t1.join();
  t2.join();
  t3.join();
  t4.join();
}

static void process(float dt, vec3 &pos, const vec3 &vel)
{
  pos += vel * dt;
}

SYSTEM(ecs::SystemTag::GameEditor) dag_ecs_update(vec3 &p, const vec3 &v)
{
  process(Time::delta_time(), p, v);
}

SYSTEM(ecs::SystemTag::GameEditor) prune_cache0()
{
  prune_cache();
}

SYSTEM(ecs::SystemTag::GameEditor) dag_vector_structs_update()
{
  for (DagorTestEntity &entity : list0)
  {
    process(Time::delta_time(), entity.p, entity.v);
  }
}

SYSTEM(ecs::SystemTag::GameEditor) prune_cache1()
{
  prune_cache();
}

SYSTEM(ecs::SystemTag::GameEditor) dag_vector_pointers_update()
{
  for (auto &entity : list1)
  {
    process(Time::delta_time(), entity->p, entity->v);
  }
}

SYSTEM(ecs::SystemTag::GameEditor) prune_cache2()
{
  prune_cache();
}

SYSTEM(ecs::SystemTag::GameEditor) dag_vector_pointers_virtual_update()
{
  for (auto &entity : list2)
  {
    entity->update(Time::delta_time());
  }
}

SYSTEM(ecs::SystemTag::GameEditor) prune_cache3()
{
  prune_cache();
}


SYSTEM(ecs::SystemTag::GameEditor) tiny_ecs_update(vec3 &P, const vec3 &V)
{
  process(Time::delta_time(), P, V);
}

SYSTEM(ecs::SystemTag::GameEditor) prune_cache4()
{
  prune_cache();
}

SYSTEM(ecs::SystemTag::GameEditor) tiny_soa_structs_update()
{
  for (uint i = 0, n = vData.size(); i < n; i++)
  {
    process(Time::delta_time(), pData[i], vData[i]);
  }
}

SYSTEM(ecs::SystemTag::GameEditor) prune_cache5()
{
  prune_cache();
}

SYSTEM(ecs::SystemTag::GameEditor) tiny_vector_structs_update()
{
  for (TinyEntity &entity : tinyList0)
  {
    process(Time::delta_time(), entity.P, entity.V);
  }
}

SYSTEM(ecs::SystemTag::GameEditor) prune_cache6()
{
  prune_cache();
}

void dag_ecs_update_func();
void dag_vector_structs_update_func();
void dag_vector_pointers_update_func();
void dag_vector_pointers_virtual_update_func();
void tiny_ecs_update_func();
void tiny_soa_structs_update_func();
void tiny_vector_structs_update_func();


void tiny_vector_and_soa_update()
{
  auto handle = std::thread([]()
  {
    dag_vector_structs_update_func();
  });

  auto handle1 = std::thread([&]()
  {
    dag_vector_pointers_update_func();
    tiny_ecs_update_func();
  });
  auto handle2 = std::thread([&]()
  {
    tiny_vector_structs_update_func();
    tiny_soa_structs_update_func();
  });
  dag_ecs_update_func();
  dag_vector_pointers_virtual_update_func();
  handle.join();
  handle1.join();
  handle2.join();
}
