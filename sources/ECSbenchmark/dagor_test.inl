#include <ecs.h>
#include <application/time.h>
#include <3dmath.h>
#include "constanta.h"
#include <thread>


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


static vector<DagorTestEntity> list0;
static vector<DagorTestEntity*> list1;
static vector<DagorTestEntity*> list2;

static vector<vec3> pData;
static vector<vec3> vData;

volatile int cache0 = 0;
void prune_cache()
{
  static vector<int> memory;
  if (!memory.size())
    memory.resize(4<<20, 1);
  for (auto i:memory)
    cache0 += i;
}

EVENT(scene=game, editor) dag_init(const ecs::OnSceneCreated &)
{

  debug_log("struct sizeof = %d", sizeof(DagorTestEntity));
  const ecs::Template *tmpl1 = ecs::create_template("test1", {
    {"p", vec3()},
    {"v", vec3()},
    {"int_variable", 10},
    {"int_component2", 10},
    {"data0", mat4(1.f)},
    {"data1", mat4(1.f)},
    {"data2", mat4(1.f)},
    {"data3", mat4(1.f)},
    {"data4", mat4(1.f)},
    {"data5", mat4(1.f)},
    {"data6", mat4(1.f)},
    {"data7", mat4(1.f)},
    {"data8", mat4(1.f)},
    {"data9", mat4(1.f)}
  });
  {
    TimeScope a("ecs_create");
    for (uint i = 0; i < dagorEntityCount; i++)
    {
      ecs::create_entity(tmpl1, {
        {"p", rand_vec3()},
        {"v", rand_vec3()}
      });
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
  uint tinyCount = dagorEntityCount;
  {
    TimeScope b("tiny_SoA_creation");
    for (uint i = 0; i < tinyCount; i++)
    {
      pData.emplace_back(rand_vec3());
      vData.emplace_back(rand_vec3());
    }
  }
  fflush(stdout);
  return;
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
SYSTEM(stage=act;scene=game, editor) cache_trach(mat4 &data0, mat4 &data1, mat4 &data2, mat4 &data3)
{
  data0 = data1;
  data1 = data2;
  data2 = data3;
  data3 += mat4(0.001f);
}
static void process(float dt, vec3 &pos, const vec3 &vel)
{
  pos += vel * dt;
}

SYSTEM(stage=act;scene=game, editor) dag_soa_update()
{
  vec3 *__restrict pos = pData.data();
  const vec3 *__restrict vel = vData.data();
  for (uint i = 0, n = vData.size(); i < n; i++)
  {
    process(Time::delta_time(), pos[i], vel[i]);
  }
}

SYSTEM(stage=act;scene=game, editor) prune_cache_()
{
  prune_cache();
}
SYSTEM(stage=act;scene=game, editor) dag_ecs_update(vec3 &p, const vec3 &v)
{
  process(Time::delta_time(), p, v);
}

SYSTEM(stage=act;scene=game, editor) prune_cache0()
{
  prune_cache();
}

SYSTEM(stage=act;scene=game, editor) dag_vector_structs_update()
{
  for (DagorTestEntity &entity : list0)
  {
    process(Time::delta_time(), entity.p, entity.v);
  }
}

SYSTEM(stage=act;scene=game, editor) prune_cache1()
{
  prune_cache();
}

SYSTEM(stage=act;scene=game, editor) dag_vector_pointers_update()
{
  for (auto &entity : list1)
  {
    process(Time::delta_time(), entity->p, entity->v);
  }
}

SYSTEM(stage=act;scene=game, editor) prune_cache2()
{
  prune_cache();
}

SYSTEM(stage=act;scene=game, editor) dag_vector_pointers_virtual_update()
{
  for (auto &entity : list2)
  {
    entity->update(Time::delta_time());
  }
}

SYSTEM(stage=act;scene=game, editor) prune_cache3()
{
  prune_cache();
}


void dag_ecs_update_func();
void dag_vector_structs_update_func();
void dag_vector_pointers_update_func();
void dag_vector_pointers_virtual_update_func();
void dag_soa_update_func();


void tiny_vector_and_soa_update()
{
  auto handle = std::thread([]()
  {
    dag_vector_structs_update_func();
  });

  auto handle1 = std::thread([&]()
  {
    dag_vector_pointers_update_func();
  });
  auto handle2 = std::thread([&]()
  {
    dag_soa_update_func();
  });
  dag_ecs_update_func();
  dag_vector_pointers_virtual_update_func();
  handle.join();
  handle1.join();
  handle2.join();
}
